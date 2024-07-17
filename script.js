const video = document.getElementById('video');
const textDisplay = document.getElementById('text-display');
const clearButton = document.getElementById('clear-button');

let socket = io();

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(error => {
        console.error('Error accessing camera:', error);
    });

const captureFrame = () => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0);

    // Extract ROI or process the frame as needed
    const frameData = canvas.toDataURL('image/jpeg');

    socket.emit('frame', frameData);
};

setInterval(captureFrame, 100); // Capture frame every 100ms

clearButton.addEventListener('click', () => {
    textDisplay.innerText = '';
    socket.emit('clear');
});

socket.on('prediction', (data) => {
    const { label, score } = data;
    console.log('Prediction:', label, score);

    if (label === 'space') {
        textDisplay.innerText += ' ';
    } else if (label === 'del') {
        textDisplay.innerText = textDisplay.innerText.slice(0, -1);
    } else {
        textDisplay.innerText += label;
    }
});
