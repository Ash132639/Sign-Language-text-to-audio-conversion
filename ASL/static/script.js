let recognizedText = '';

function addCharacter() {
    const currentPrediction = document.querySelector('#video').getAttribute('data-prediction');
    if (currentPrediction === 'Space') {
        recognizedText += ' ';
    } else {
        recognizedText += currentPrediction;
    }
    updateTextarea();
}

function removeCharacter() {
    recognizedText = recognizedText.slice(0, -1);
    updateTextarea();
}

function updateTextarea() {
    document.querySelector('#recognized-text').value = recognizedText;
}