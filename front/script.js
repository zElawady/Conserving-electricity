const API_BASE = `http://${window.location.hostname}:${window.location.port}/api`;
let currentColumns = [];
let selectedModelId = null;
let currentTaskType = "Classification"; // Default to Classification

// Models that only work for classification
const CLASSIFICATION_ONLY = new Set(["LogisticRegression", "NaiveBayes", "ANN"]);
// Models that only work for regression
const REGRESSION_ONLY = new Set(["LinearRegression"]);
// Models compatible with both classification and regression
const BOTH_TASKS = new Set(["DecisionTree", "RandomForest", "SVM", "KNN"]);
// Models for clustering
const CLUSTERING_ONLY = new Set(["kmeans"]);

function updateTaskType(val) {
    currentTaskType = val;
    // Sync both selectors
    const s1 = document.getElementById('manual-task-type');
    const s2 = document.getElementById('model-task-type');
    if (s1) s1.value = val;
    if (s2) s2.value = val;

    filterModelCards(currentTaskType);
    const info = document.getElementById('task-status-info');
    if (info) {
        info.innerHTML = `<ion-icon name="hand-right-outline" style="margin-right: 0.4rem; font-size: 1.2rem; color: #fbbf24;"></ion-icon> Task type set manually`;
        info.style.color = "#fbbf24";
    }
}

function filterModelCards(taskType) {
    document.querySelectorAll('.algo-card').forEach(card => {
        const modelId = card.getAttribute('onclick').match(/'([^']+)'\)/)?.[1];
        if (!modelId || !taskType) {
            card.style.display = 'block';
            card.classList.remove('dimmed', 'incompatible');
            return;
        }
        const isClassification = taskType === "Classification";
        const isRegression = taskType === "Regression";
        const isClustering = taskType === "Clustering";

        let compatible = false;
        if (isClustering) {
            compatible = CLUSTERING_ONLY.has(modelId);
        } else {
            if (BOTH_TASKS.has(modelId)) {
                compatible = true;
            } else if (isClassification && CLASSIFICATION_ONLY.has(modelId)) {
                compatible = true;
            } else if (isRegression && REGRESSION_ONLY.has(modelId)) {
                compatible = true;
            }
        }

        if (compatible) {
            card.style.display = 'block';
            card.classList.remove('dimmed', 'incompatible');
        } else {
            // ALL OPEN: No more dimming or hiding
            card.style.display = 'block';
            card.classList.remove('dimmed', 'incompatible');
        }
    });

    // Update section labels (hide if no models under them)
    document.querySelectorAll('.section-label').forEach(label => {
        const grid = label.nextElementSibling;
        if (grid && grid.classList.contains('model-grid')) {
            const hasVisible = Array.from(grid.children).some(child => child.style.display !== 'none');
            label.style.display = hasVisible ? 'block' : 'none';
        }
    });

    const badge = document.getElementById('task-type-badge');
    if (badge) {
        badge.textContent = taskType ? `Detected Task: ${taskType}` : '';
        badge.style.background = taskType === 'Classification' ? 'rgba(37,99,235,0.25)' :
            taskType === 'Regression' ? 'rgba(217,119,6,0.25)' : 'transparent';
        badge.style.color = taskType === 'Classification' ? '#60a5fa' :
            taskType === 'Regression' ? '#fbbf24' : 'transparent';
    }
}

// Re-detect task type whenever the target column changes
const modelTargetSelect = document.getElementById('model-target');
if (modelTargetSelect) {
    modelTargetSelect.addEventListener('change', async () => {
        const target = modelTargetSelect.value;
        if (!target) return;
        try {
            const res = await fetch(`${API_BASE}/detect_task?target=${encodeURIComponent(target)}`);
            if (res.ok) {
                const data = await res.json();
                currentTaskType = data.task_type;
                filterModelCards(currentTaskType);
            }
        } catch (_) { /* silently ignore */ }
    });
}

document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.page).classList.add('active');
    });
});

function updateDropdowns(columns) {
    currentColumns = columns;
    document.querySelectorAll('.dynamic-cols').forEach(select => {
        const val = select.value;
        select.innerHTML = columns.map(col => `<option value="${col}">${col}</option>`).join('');
        if (columns.includes(val)) select.value = val;
    });
}

// 1. Upload
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
if (dropZone && fileInput) {
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('dragover'); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault(); dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) uploadFile(e.target.files[0]);
    });
}

async function uploadFile(file) {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try {
        const btn = document.querySelector('.btn-upload');
        btn.innerText = "UPLOADING..."; btn.disabled = true;
        const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
        const data = await res.json();
        btn.innerText = "BROWSE COMPUTER"; btn.disabled = false;
        if (!res.ok) throw new Error(data.detail);
        updateDropdowns(data.columns);
        // Store the detected task type and immediately filter model cards
        currentTaskType = data.task_guess || "Classification";

        // Sync selectors
        const s1 = document.getElementById('manual-task-type');
        const s2 = document.getElementById('model-task-type');
        if (s1) s1.value = currentTaskType;
        if (s2) s2.value = currentTaskType;

        filterModelCards(currentTaskType);
        document.getElementById('upload-report').classList.remove('hidden');
        document.getElementById('report-summary').innerHTML = `<b>Dimensions:</b> ${data.shape[0]} Rows x ${data.shape[1]} Columns | <b>Task:</b> ${currentTaskType}`;
        document.getElementById('num-cols').innerHTML = data.numerical_columns.map(c => `<span class="badge" style="background:#2563eb">${c}</span>`).join('');
        document.getElementById('cat-cols').innerHTML = data.categorical_columns.map(c => `<span class="badge" style="background:#d97706">${c}</span>`).join('');
        const missingHtml = Object.entries(data.missing_values || {}).map(([col, val]) => `<span class="badge" style="background:${val > 0 ? '#dc2626' : '#059669'}">${col}: ${val}</span>`).join('');
        document.getElementById('missing-vals').innerHTML = missingHtml;
        document.getElementById('data-preview').innerHTML = `<thead><tr>${Object.keys(data.head[0]).map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>${data.head.map(r => `<tr>${Object.values(r).map(v => `<td>${v}</td>`).join('')}</tr>`).join('')}</tbody>`;
    } catch (e) { alert(e.message); }
}

// 2. Visualization
async function generatePlot() {
    const type = document.getElementById('viz-type').value;
    const x = document.getElementById('viz-x').value;
    const y = document.getElementById('viz-y').value;
    try {
        const res = await fetch(`${API_BASE}/visualization/${type}?x=${x}&y=${y}`);
        const data = await res.json();
        const baseUrl = `http://${window.location.hostname}:${window.location.port}`;
        document.getElementById('plot-result').innerHTML = `<div class="plot-item"><img src="${baseUrl}${data.plot_url}?t=${new Date().getTime()}" /></div>`;
    } catch (e) { alert(e.message); }
}

// 3. Preprocessing
async function applyPreprocessing(methodId, colId) {
    const action = document.getElementById(methodId).value;
    const col = colId ? document.getElementById(colId).value : null;
    try {
        const res = await fetch(`${API_BASE}/preprocess`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, column: col })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);
        updateDropdowns(data.columns);
        document.getElementById('prep-status').innerHTML = `
            <div style="animation: fadeIn 0.4s ease;">
                <h2 style="color:var(--success)">✅ Applied ${action.replace('_', ' ').toUpperCase()}</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top:1rem;">
                    <div class="data-table-container"><table class="data-table"><thead><tr>${Object.keys(data.before_head[0]).map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>${data.before_head.map(r => `<tr>${Object.values(r).map(v => `<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
                    <div class="data-table-container"><table class="data-table"><thead><tr>${Object.keys(data.after_head[0]).map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>${data.after_head.map(r => `<tr>${Object.values(r).map(v => `<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
                </div>
            </div>
        `;
    } catch (e) { alert(e.message); }
}

function selectModel(element, modelId) {
    // 1. Remove active class from previous selection
    document.querySelectorAll('.algo-card').forEach(c => c.classList.remove('active'));

    // 2. Add active class to current selection
    element.classList.add('active');

    // 3. Store selected model ID
    selectedModelId = modelId;

    // Toggle target selection visibility for K-Means
    const targetLabel = document.getElementById('target-label');
    const targetSelect = document.getElementById('model-target');
    if (modelId === 'kmeans') {
        if (targetLabel) targetLabel.style.display = 'none';
        if (targetSelect) targetSelect.style.display = 'none';
        selectedModelId = 'kmeans';
    } else {
        if (targetLabel) targetLabel.style.display = 'block';
        if (targetSelect) targetSelect.style.display = 'block';
    }

    const info = document.getElementById('selected-model-info');
    const name = element.querySelector('p').innerText;
    
    // Hint about target requirement
    const targetHint = modelId === 'kmeans' ? '<p style="color:#fbbf24; font-size:0.75rem; margin-top:0.5rem;">Unsupervised Mode: No target column needed.</p>' : '';
    
    info.innerHTML = `
        <div style="animation: fadeIn 0.3s ease;">
            <ion-icon name="checkmark-circle" style="font-size:2.5rem; color:#10b981; margin-bottom:0.5rem;"></ion-icon>
            <h3 style="color:#fff; font-size:1rem;">${name}</h3>
            <p style="color:var(--success); font-size:0.85rem;">Selected & Ready</p>
            ${targetHint}
        </div>`;
}

async function trainSelectedModel() {
    const target = document.getElementById('model-target').value;
    const isKmeans = selectedModelId === 'kmeans';
    if (!target && !isKmeans) return alert("Please select a target column first.");
    if (!selectedModelId) return alert("Please click a model card to select it.");

    // Frontend mismatch guard removed to allow user freedom

    try {
        const report = document.getElementById('train-report');
        const loader = document.getElementById('loading-indicator');
        const evalTable = document.getElementById('eval-table');
        const cmContainer = document.getElementById('cm-container');

        report.classList.remove('hidden');
        loader.classList.remove('hidden');
        evalTable.innerHTML = '';
        cmContainer.innerHTML = '';
        report.scrollIntoView({ behavior: 'smooth' });

        // FIX: Sending proper POST request with body
        const res = await fetch(`${API_BASE}/train_single`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target: target, algo: selectedModelId, task_type: currentTaskType })
        });
        const data = await res.json();
        loader.classList.add('hidden');

        if (!res.ok) throw new Error(data.detail);

        // Display results
        const metrics = data.metrics;
        const rows = Object.keys(metrics).filter(k => k !== 'confusion_matrix' && k !== 'classification_report');

        evalTable.innerHTML = `
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
                ${rows.map(r => {
            let val = metrics[r];
            if (typeof val === 'number') {
                // High precision for small errors, standard 4 for scores
                val = (r === 'mse' || r === 'rmse' || r === 'mae') ? val.toFixed(6) : val.toFixed(4);
            }
            return `<tr><td>${r.toUpperCase()}</td><td>${val}</td></tr>`;
        }).join('')}
            </tbody>
        `;

        if (data.cm_url) {
            const baseUrl = `http://${window.location.hostname}:${window.location.port}`;
            const plotTitle = selectedModelId === 'kmeans' ? 'Cluster Visualization' : 'Confusion Matrix';
            cmContainer.innerHTML = `
                <h4 style="margin-bottom:1rem; color:#fff;">${plotTitle}</h4>
                <img src="${baseUrl}${data.cm_url}?t=${new Date().getTime()}" style="max-width:500px; border-radius:0.5rem; border:1px solid var(--border);" />
            `;
        }
    } catch (e) {
        document.getElementById('loading-indicator').classList.add('hidden');
        let errorMsg = e.message;
        if (errorMsg.includes("continuous")) {
            errorMsg = "Task Mismatch: You are trying to use a CLASSIFACTION model on REGRESSION data (continuous values). Please select a Regression model instead.";
        }
        alert("Training Error: " + errorMsg);
    }
}

// Initialize
filterModelCards(currentTaskType);
