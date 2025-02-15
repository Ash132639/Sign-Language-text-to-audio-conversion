let detectedText = '';

function addToText() {
    const prediction = document.getElementById('current-prediction').textContent;
    if (prediction) {
        detectedText += prediction + ' ';
        document.getElementById('detected-text').value = detectedText;
    }
}

function clearText() {
    detectedText = '';
    document.getElementById('detected-text').value = '';
}

function speakText() {
    const text = document.getElementById('detected-text').value;
    if (text) {
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    }
}