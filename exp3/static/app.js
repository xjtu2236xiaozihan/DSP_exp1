// app.js

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. 获取 DOM 元素 ---
    const orb = document.getElementById('orb');
    const statusLabel = document.getElementById('status-label');
    const interactionCore = document.getElementById('interaction-core');
    const resultDisplay = document.getElementById('result-display');
    
    // Orb 内部状态
    const iconIdle = document.getElementById('icon-idle');
    const visualizerCanvas = document.getElementById('visualizer');
    const loader = document.getElementById('loader');

    // 结果
    const resultDigit = document.getElementById('result-digit');
    const dtwDistance = document.getElementById('dtw-distance');
    const resetButton = document.getElementById('reset-button');

    // --- 2. Web Audio API & 录音设置 ---
    let mediaRecorder;
    let audioChunks = [];
    let audioContext;
    let visualizerContext = visualizerCanvas.getContext('2d');
    let analyser;
    let animationFrameId;

    // --- 3. 状态管理 ---
    const states = {
        IDLE: 'IDLE',
        RECORDING: 'RECORDING',
        PROCESSING: 'PROCESSING',
        RESULT: 'RESULT'
    };
    let currentState = states.IDLE;

    function setState(newState) {
        currentState = newState;
        
        // 隐藏所有 Orb 状态
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
                startAudioCapture();
                break;

            case states.PROCESSING:
                statusLabel.textContent = '分析中...DTW 匹配中...';
                loader.classList.remove('hidden');
                stopAudioCapture();
                break;

            case states.RESULT:
                interactionCore.classList.add('hidden');
                resultDisplay.classList.remove('hidden');
                break;
        }
    }

    // --- 4. 核心交互事件 ---
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

    // --- 5. 音频捕获与可视化 ---
    async function startAudioCapture() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('您的浏览器不支持麦克风访问！');
            setState(states.IDLE);
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = []; // 清空之前的录音

            // 可视化
            setupVisualizer(stream);
            
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                sendAudioToBackend(audioBlob);
                stream.getTracks().forEach(track => track.stop()); // 关闭麦克风
                cancelAnimationFrame(animationFrameId); // 停止动画
            };

            mediaRecorder.start();
        } catch (err) {
            console.error("麦克风访问失败:", err);
            alert("无法访问麦克风。");
            setState(states.IDLE);
        }
    }

    function stopAudioCapture() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        }
    }
    
    // --- 6. 声波可视化 (Canvas) ---
    function setupVisualizer(stream) {
        if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        visualizerContext.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
        
        const draw = () => {
            animationFrameId = requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray); // 获取波形数据

            visualizerContext.fillStyle = 'rgba(10, 10, 26, 0.1)'; // 清除画布（带拖影效果）
            visualizerContext.fillRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
            
            visualizerContext.lineWidth = 2;
            visualizerContext.strokeStyle = 'var(--color-secondary-accent)'; // 紫色波形
            visualizerContext.beginPath();

            const sliceWidth = visualizerCanvas.width * 1.0 / bufferLength;
            let x = 0;

            for(let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0; // 归一化到 0-2
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

    // --- 7. 后端通信 ---
    async function sendAudioToBackend(audioBlob) {
        const formData = new FormData();
        formData.append('audio_file', audioBlob, 'user_digit.wav');

        try {
            // !! 假设你有一个运行在 5000 端口的后端服务 !!
            const response = await fetch('/recognize', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`服务器错误: ${response.statusText}`);
            }

            const data = await response.json();

            // 更新结果
            resultDigit.textContent = data.digit;
            dtwDistance.textContent = data.distance.toFixed(2);
            setState(states.RESULT);

        } catch (error) {
            console.error('识别失败:', error);
            alert(`识别失败: ${error.message}`);
            setState(states.IDLE);
        }
    }

    // 初始化
    setState(states.IDLE);
});