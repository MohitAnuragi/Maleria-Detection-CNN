const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const loading = document.getElementById('loading');
const resultBox = document.getElementById('resultBox');

// UI Elements for result
const previewImage = document.getElementById('previewImage');
const predictionText = document.getElementById('predictionText');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');

// Setup Drag & Drop
uploadBox.addEventListener('click', () => fileInput.click());

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert("Please upload a valid image file.");
        return;
    }

    // Show preview immediately
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Switch UI to loading
    uploadBox.classList.add('hidden');
    loading.classList.remove('hidden');

    // Send to backend
    const formData = new FormData();
    formData.append('image', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            resetUI();
            return;
        }
        showResult(data);
    })
    .catch(error => {
        console.error('Error:', error);
        alert("An error occurred during prediction. Is the server running?");
        resetUI();
    });
}

function showResult(data) {
    loading.classList.add('hidden');
    resultBox.classList.remove('hidden');
    
    // Set text and colors
    predictionText.textContent = data.prediction;
    predictionText.className = data.prediction.toLowerCase();
    
    // Animate confidence bar
    const confPercent = (data.confidence * 100).toFixed(2);
    confidenceText.textContent = `Confidence: ${confPercent}%`;
    
    // Small delay to allow CSS transition to work after unhiding
    setTimeout(() => {
        confidenceBar.style.width = `${confPercent}%`;
        
        // Color bar based on result
        if (data.prediction === 'Parasitized') {
            confidenceBar.style.background = 'linear-gradient(90deg, #F87171, #EF4444)';
        } else {
            confidenceBar.style.background = 'linear-gradient(90deg, #34D399, #10B981)';
        }
    }, 50);
}

function resetUI() {
    resultBox.classList.add('hidden');
    loading.classList.add('hidden');
    uploadBox.classList.remove('hidden');
    fileInput.value = ''; // clear input
    confidenceBar.style.width = '0%'; // reset bar
}
