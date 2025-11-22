// exp3/static/app.js - 修复版 (原生 WAV 录制)

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. DOM 元素 ---
    const orb = document.getElementById('orb');
    const statusLabel = document.getElementById('status-label');
    const interactionCore = document.getElementById('interaction-core');
    const resultDisplay = document.getElementById('result-display');
    
    // Orb 内部
    const iconIdle = document.getElementById('icon-idle');
    const visualizerCanvas = document.getElementById('visualizer');
    const loader = document.getElementById('loader');

    // 结果
    const resultDigit = document.getElementById('result-digit');
    const dtwDistance = document.getElementById('dtw-distance');
    const resetButton = document.getElementById('reset-button');

    // --- 2. 状态定义 ---
    const states = {
        IDLE: 'IDLE',
        RECORDING: 'RECORDING',
        PROCESSING: 'PROCESSING',
        RESULT: 'RESULT'
    };
    let currentState = states.IDLE;

    // --- 3. 音频处理变量 ---
    let audioContext = null;
    let mediaStream = null;
    let scriptProcessor = null;
    let audioInput = null;
    let audioBuffers = []; // 存储 PCM 数据
    let bufferLength = 0;
    
    // 可视化
    let visualizerContext = visualizerCanvas.getContext('2d');
    let analyser = null;
    let animationFrameId = null;

    // --- 4. 状态切换逻辑 ---
    function setState(newState) {
        currentState = newState;
        
        // UI 重置
        [iconIdle, visualizerCanvas, loader].forEach(el => el.classList.add('hidden'));
        orb.classList.remove('recording');

        switch(newState) {
            case states.IDLE:
                statusLabel.textContent = '点击以激活';
                iconIdle.classList.remove('hidden');
                interactionCore.classList.remove('hidden');
                resultDisplay.classList.add('hidden');
                break;
            
            case states.RECORDING:
                statusLabel.textContent = '正在聆听... (点击停止)';
                visualizerCanvas.classList.remove('hidden');
                orb.classList.add('recording');
                startWavRecording(); // <--- 使用新的录音函数
                break;

            case states.PROCESSING:
                statusLabel.textContent = '分析中...';
                loader.classList.remove('hidden');
                stopWavRecording(); // <--- 使用新的停止函数
                break;

            case states.RESULT:
                interactionCore.classList.add('hidden');
                resultDisplay.classList.remove('hidden');
                break;
        }
    }

    // --- 5. 交互事件 ---
    orb.addEventListener('click', () => {
        if (currentState === states.IDLE) {
            setState(states.RECORDING);
        } else if (currentState === states.RECORDING) {
            setState(states.PROCESSING);
        }
    });

    resetButton.addEventListener('click', () => {
        setState(states.IDLE);
        resultDigit.textContent = '_';
        dtwDistance.textContent = '---';
    });

    // ==========================================
    // [核心修复] 原生 WAV 录音实现 (替代 MediaRecorder)
    // ==========================================
    
    async function startWavRecording() {
        try {
            // 初始化 AudioContext
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 }); // 强制 16kHz
            
            // 获取麦克风流
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // 创建源
            audioInput = audioContext.createMediaStreamSource(mediaStream);
            
            // 创建处理节点 (Buffer Size 4096, 1 Input, 1 Output)
            //虽然 ScriptProcessor 已废弃，但它是目前无需外部文件(AudioWorklet)最简单的实现方式
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
            
            // 重置缓冲区
            audioBuffers = [];
            bufferLength = 0;

            // 监听音频数据
            scriptProcessor.onaudioprocess = (event) => {
                const inputBuffer = event.inputBuffer.getChannelData(0); // 获取单声道数据
                // 深拷贝一份数据存起来
                const bufferCopy = new Float32Array(inputBuffer);
                audioBuffers.push(bufferCopy);
                bufferLength += bufferCopy.length;
            };

            // 连接节点: Source -> Processor -> Destination
            audioInput.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);

            // 启动可视化
            setupVisualizer(mediaStream);

        } catch (err) {
            console.error("录音启动失败:", err);
            alert("无法访问麦克风: " + err.message);
            setState(states.IDLE);
        }
    }

    function stopWavRecording() {
        if (scriptProcessor && audioInput) {
            // 断开连接
            audioInput.disconnect();
            scriptProcessor.disconnect();
            
            // 停止流
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
            }

            // 合并 buffer 并生成 WAV Blob
            const wavBlob = exportWAV(audioBuffers, bufferLength, audioContext.sampleRate);
            
            // 发送给后端
            sendAudioToBackend(wavBlob);
        }
        
        // 停止动画
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
    }

    // --- WAV 编码核心函数 (将 Float32 转换为 16-bit PCM WAV) ---
    function exportWAV(buffers, totalLength, sampleRate) {
        // 1. 合并所有 Buffer
        const mergedBuffer = new Float32Array(totalLength);
        let offset = 0;
        for (let i = 0; i < buffers.length; i++) {
            mergedBuffer.set(buffers[i], offset);
            offset += buffers[i].length;
        }

        // 2. 创建 WAV 文件头 (44 bytes)
        const buffer = new ArrayBuffer(44 + mergedBuffer.length * 2);
        const view = new DataView(buffer);

        // RIFF chunk descriptor
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + mergedBuffer.length * 2, true);
        writeString(view, 8, 'WAVE');
        
        // fmt sub-chunk
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);   // Subchunk1Size (16 for PCM)
        view.setUint16(20, 1, true);    // AudioFormat (1 for PCM)
        view.setUint16(22, 1, true);    // NumChannels (1 for Mono)
        view.setUint32(24, sampleRate, true); // SampleRate
        view.setUint32(28, sampleRate * 2, true); // ByteRate
        view.setUint16(32, 2, true);    // BlockAlign
        view.setUint16(34, 16, true);   // BitsPerSample

        // data sub-chunk
        writeString(view, 36, 'data');
        view.setUint32(40, mergedBuffer.length * 2, true);

        // 3. 写入 PCM 数据 (Float32 -> Int16)
        floatTo16BitPCM(view, 44, mergedBuffer);

        return new Blob([view], { type: 'audio/wav' });
    }

    function floatTo16BitPCM(output, offset, input) {
        for (let i = 0; i < input.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, input[i])); // Clamp
            // Convert to 16-bit PCM
            s = s < 0 ? s * 0x8000 : s * 0x7FFF;
            output.setInt16(offset, s, true);
        }
    }

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // --- 6. 后端通信 (保持不变) ---
    async function sendAudioToBackend(audioBlob) {
        const formData = new FormData();
        formData.append('audio_file', audioBlob, 'mic_record.wav'); // 现在是真正的 wav

        try {
            const response = await fetch('/recognize', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                // 尝试解析错误信息
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `HTTP ${response.status}`);
            }

            const data = await response.json();
            resultDigit.textContent = data.digit;
            dtwDistance.textContent = data.distance.toFixed(2);
            setState(states.RESULT);

        } catch (error) {
            console.error('识别失败:', error);
            alert(`识别失败: ${error.message}`);
            setState(states.IDLE);
        }
    }

    // --- 7. 可视化 (复用原有逻辑，微调) ---
    function setupVisualizer(stream) {
        if (!audioContext) return;
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        analyser.fftSize = 256;
        const bufferLen = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLen);
        
        const draw = () => {
            animationFrameId = requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray);

            visualizerContext.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
            visualizerContext.fillStyle = 'rgba(10, 10, 26, 0.1)'; 
            visualizerContext.fillRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
            
            visualizerContext.lineWidth = 2;
            visualizerContext.strokeStyle = '#00f3ff'; // 青色波形
            visualizerContext.beginPath();

            const sliceWidth = visualizerCanvas.width * 1.0 / bufferLen;
            let x = 0;

            for(let i = 0; i < bufferLen; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * visualizerCanvas.height / 2;
                if(i === 0) visualizerContext.moveTo(x, y);
                else visualizerContext.lineTo(x, y);
                x += sliceWidth;
            }
            visualizerContext.lineTo(visualizerCanvas.width, visualizerCanvas.height / 2);
            visualizerContext.stroke();
        };
        draw();
    }
});