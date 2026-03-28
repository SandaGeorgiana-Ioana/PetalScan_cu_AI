const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const predictBtn = document.getElementById('predictBtn');
const resultSection = document.getElementById('resultSection');
const resultMessage = document.getElementById('resultMessage');
const resultsList = document.getElementById('resultsList');
const loading = document.getElementById('loading');

// Click pe upload box
uploadBox.addEventListener('click', () => fileInput.click());

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#e91e8c';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#f48fb1';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#f48fb1';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        loadImage(file);
    }
});

// Selectare fișier
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) loadImage(file);
});

// Încarcă imaginea în preview
function loadImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.hidden = false;
        uploadPlaceholder.hidden = true;
        predictBtn.disabled = false;
        resultSection.hidden = true;
    };
    reader.readAsDataURL(file);
}

// Buton predicție
predictBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    // Arată loading
    loading.hidden = false;
    resultSection.hidden = true;
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            showError(data.error);
            return;
        }

        showResults(data);

    } catch (err) {
        showError('Eroare de conexiune!');
    } finally {
        loading.hidden = true;
        predictBtn.disabled = false;
    }
});

// Afișează rezultatele
function showResults(data) {
    // Mesaj principal
    resultMessage.textContent = data.mesaj;
    resultMessage.className = 'result-message ' + (data.este_floare ? 'succes' : 'nesigur');

    // Lista top 5
    resultsList.innerHTML = '';
    data.rezultate.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'result-item';
        div.innerHTML = `
            <span class="result-name">${index + 1}. ${item.floare}</span>
            <div class="result-bar-container">
                <div class="result-bar" style="width: ${item.probabilitate}%"></div>
            </div>
            <span class="result-prob">${item.probabilitate}%</span>
        `;
        resultsList.appendChild(div);
    });

    resultSection.hidden = false;
}

// Afișează eroare
function showError(msg) {
    resultMessage.textContent = '❌ ' + msg;
    resultMessage.className = 'result-message nesigur';
    resultsList.innerHTML = '';
    resultSection.hidden = false;
}
