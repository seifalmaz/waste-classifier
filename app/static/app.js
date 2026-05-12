let currentStream = null;
let resultsInterval = null;
let statsChart = null;

// =====================================================
// NAVIGATION
// =====================================================

function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.style.display = 'none');
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    
    // Show target
    document.getElementById(pageId).style.display = 'block';
    event.currentTarget.classList.add('active');

    // Special logic for pages
    if (pageId === 'realtime') {
        startStream();
    } else {
        stopStream();
    }

    if (pageId === 'analytics') {
        renderCharts();
    }
}

// =====================================================
// REAL-TIME LOGIC
// =====================================================

function startStream() {
    const streamImg = document.getElementById('stream');
    streamImg.src = '/video_feed';
    
    // Poll for results
    resultsInterval = setInterval(async () => {
        try {
            const resp = await fetch('/results');
            const data = await resp.json();
            updateRealtimeUI(data);
        } catch (e) { console.error(e); }
    }, 200);
}

function stopStream() {
    const streamImg = document.getElementById('stream');
    streamImg.src = '';
    fetch('/stop_feed');
    if (resultsInterval) clearInterval(resultsInterval);
}

function updateRealtimeUI(data) {
    const resDiv = document.getElementById('detection-result');
    const confBar = document.getElementById('conf-bar');
    const confText = document.getElementById('conf-text');
    const fpsVal = document.getElementById('fps-val');
    const streamFps = document.getElementById('stream-fps');
    const totalCount = document.getElementById('total-count');

    if (data.detections && data.detections.length > 0) {
        const best = data.detections[0];
        const confPerc = formatConfidence(best.confidence);
        
        resDiv.innerText = best.class;
        resDiv.style.color = getCategoryColor(best.class);
        
        confBar.style.width = confPerc + '%';
        confBar.style.backgroundColor = getCategoryColor(best.class);
        confText.innerText = confPerc + '% Confidence';
        
        // Pass the raw confidence to the history manager
        addToHistory(best.class, best.confidence);
    } else {
        // Send heartbeat to allow timeout reset
        addToHistory("Unknown", 0);
        
        resDiv.innerText = "Scanning...";
        resDiv.style.color = "var(--text-secondary)";
        confBar.style.width = '0%';
        confText.innerText = '0% Confidence';
    }

    fpsVal.innerText = data.fps + ' FPS';
    streamFps.innerText = data.fps;
    
    if (data.stats) {
        const total = Object.values(data.stats).reduce((a, b) => a + b, 0);
        totalCount.innerText = total;
    }
}

// =====================================================
// STATEFUL DETECTION ENGINE (NO-SPAM)
// =====================================================

const DETECTION_CONFIG = {
    minConfidence: 0.70,      // Gate for history logging
    stabilityRequired: 3,     // Frames needed to commit
    disappearTimeoutMs: 2500, // Time before resetting state
    maxEntries: 50
};

let history = [];
let detectionState = {
    activeLabel: '',          // Currently stable visible class
    lastCommittedLabel: '',   // To prevent duplicates
    lastSeenTime: 0,          // Timestamp heartbeat
    stabilityCounter: 0,
    pendingLabel: ''
};

function addToHistory(label, conf) {
    const now = Date.now();

    // 1. Handle Disappearance or Low Confidence
    if (label === "Unknown" || conf < DETECTION_CONFIG.minConfidence) {
        if (now - detectionState.lastSeenTime > DETECTION_CONFIG.disappearTimeoutMs) {
            detectionState.lastCommittedLabel = ''; // Reset memory
            detectionState.activeLabel = '';
        }
        detectionState.stabilityCounter = 0;
        return;
    }

    // 2. Heartbeat update
    detectionState.lastSeenTime = now;

    // 3. Stability Check (Anti-Flicker)
    if (label === detectionState.pendingLabel) {
        detectionState.stabilityCounter++;
    } else {
        detectionState.pendingLabel = label;
        detectionState.stabilityCounter = 1;
    }

    if (detectionState.stabilityCounter < DETECTION_CONFIG.stabilityRequired) return;

    // 4. Duplicate Suppression (The "Only Log Once" Logic)
    if (label !== detectionState.lastCommittedLabel) {
        const entry = { 
            label, 
            conf: formatConfidence(conf), 
            time: new Date().toLocaleTimeString() 
        };
        
        history.unshift(entry);
        if (history.length > DETECTION_CONFIG.maxEntries) history.pop();
        
        detectionState.lastCommittedLabel = label;
        renderHistory();
    }
}

// Fix 10000% bug and ensure valid percentage range
function formatConfidence(val) {
    // If value is > 1, assume it's already a percentage (0-100)
    let norm = val > 1.05 ? val / 100 : val;
    // Clamp to 0-1
    norm = Math.max(0, Math.min(1, norm));
    return Math.round(norm * 100);
}

function renderHistory() {
    const list = document.getElementById('history-list');
    // Use a DocumentFragment or simple join for performance
    list.innerHTML = history.map(item => `
        <div class="history-entry" style="display: flex; justify-content: space-between; padding: 12px; border-bottom: 1px solid var(--glass-border); animation: fadeIn 0.3s ease;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background: ${getCategoryColor(item.label)}"></div>
                <span style="font-weight: 600; color: ${getCategoryColor(item.label)}">${item.label}</span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 12px; font-weight: 700;">${item.conf}%</div>
                <div style="font-size: 10px; color: var(--text-secondary)">${item.time}</div>
            </div>
        </div>
    `).join('');
}

// =====================================================
// UPLOAD LOGIC
// =====================================================

// =====================================================
// UPLOAD LOGIC (STATEFUL & RESPONSIVE)
// =====================================================

function clearUpload(event) {
    if (event) event.stopPropagation();
    
    const preview = document.getElementById('upload-preview');
    const wrapper = document.getElementById('preview-wrapper');
    const dropZone = document.getElementById('drop-zone');
    const resultsDiv = document.getElementById('upload-results');
    const fileInput = document.getElementById('file-input');

    // Reset UI State
    preview.src = '';
    preview.style.display = 'none';
    wrapper.style.display = 'none';
    dropZone.style.display = 'block';
    resultsDiv.innerHTML = '<p style="color: var(--text-secondary)">Awaiting input analysis...</p>';
    fileInput.value = ''; // Clear file selection
}

async function handleFile(file) {
    if (!file) return;

    const preview = document.getElementById('upload-preview');
    const wrapper = document.getElementById('preview-wrapper');
    const dropZone = document.getElementById('drop-zone');
    const resultsDiv = document.getElementById('upload-results');
    const loader = document.getElementById('loading-spinner');

    // Show initial preview
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        wrapper.style.display = 'block';
        dropZone.style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Process
    loader.style.display = 'block';
    resultsDiv.innerHTML = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch('/upload', { method: 'POST', body: formData });
        const data = await resp.json();
        
        // Update preview with annotated image
        preview.src = data.image;
        
        // Show results
        loader.style.display = 'none';
        if (data.results.length === 0) {
            resultsDiv.innerHTML = '<p>No waste detected in image.</p>';
        } else {
            resultsDiv.innerHTML = data.results.map(res => `
                <div class="glass-card" style="margin-bottom: 15px; background: rgba(255,255,255,0.03)">
                    <div style="font-weight: 800; font-size: 18px; color: ${getCategoryColor(res.class)}">${res.class}</div>
                    <div style="font-size: 14px; margin-top: 5px;">Confidence: ${Math.round(res.confidence * 100)}%</div>
                    <div class="conf-bar-container">
                        <div class="conf-bar-fill" style="width: ${res.confidence * 100}%; background: ${getCategoryColor(res.class)}"></div>
                    </div>
                </div>
            `).join('');
        }
    } catch (e) {
        loader.style.display = 'none';
        resultsDiv.innerHTML = '<p style="color: #ff4444">Error analyzing image.</p>';
        console.error(e);
    }
}

// =====================================================
// ANALYTICS
// =====================================================

// =====================================================
// ANALYTICS & STATS ENGINE
// =====================================================

let barChart = null;
let startTime = Date.now();

// Session Timer
setInterval(() => {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const hrs = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const mins = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    const timeDisplay = document.getElementById('session-time');
    if (timeDisplay) timeDisplay.innerText = `Session: ${hrs}:${mins}:${secs}`;
}, 1000);

async function renderCharts() {
    try {
        const resp = await fetch('/results');
        const data = await resp.json();
        const stats = data.stats || {};
        
        // 1. Calculate KPIs
        const labels = Object.keys(stats);
        const values = Object.values(stats);
        const total = values.reduce((a, b) => a + b, 0);
        
        let topCat = "None";
        let maxVal = -1;
        labels.forEach((l, i) => {
            if (values[i] > maxVal) {
                maxVal = values[i];
                topCat = l;
            }
        });

        // Update KPI Cards
        document.getElementById('stats-total').innerText = total;
        document.getElementById('stats-top').innerText = topCat;
        
        // Use history for avg confidence
        const avgConf = history.length > 0 
            ? Math.round(history.reduce((a, b) => a + b.conf, 0) / history.length) 
            : 0;
        document.getElementById('stats-conf').innerText = `${avgConf}%`;

        // 2. Render Doughnut Chart
        const distCtx = document.getElementById('dist-chart').getContext('2d');
        if (statsChart) statsChart.destroy();
        statsChart = new Chart(distCtx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: labels.map(l => getCategoryColor(l)),
                    borderWidth: 0,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right', labels: { color: '#a0a0a0', font: { size: 11 } } }
                },
                cutout: '70%'
            }
        });

        // 3. Render Horizontal Bar Chart
        const barCtx = document.getElementById('bar-chart').getContext('2d');
        if (barChart) barChart.destroy();
        barChart = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Count',
                    data: values,
                    backgroundColor: labels.map(l => getCategoryColor(l) + '55'),
                    borderColor: labels.map(l => getCategoryColor(l)),
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#a0a0a0' } },
                    y: { grid: { display: false }, ticks: { color: '#a0a0a0' } }
                },
                plugins: { legend: { display: false } }
            }
        });

        // 4. Update Activity Log
        const log = document.getElementById('analytics-history');
        if (history.length > 0) {
            log.innerHTML = history.map(item => `
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 12px; background: rgba(255,255,255,0.02); border-radius: 10px; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="width: 10px; height: 10px; border-radius: 50%; background: ${getCategoryColor(item.label)}"></div>
                        <div>
                            <div style="font-weight: 600; font-size: 14px;">${item.label}</div>
                            <div style="font-size: 11px; color: var(--text-secondary)">Verified at ${item.time}</div>
                        </div>
                    </div>
                    <div style="text-align: right; font-weight: 800; color: var(--accent-color)">${item.conf}%</div>
                </div>
            `).join('');
        }
    } catch (e) {
        console.error("Analytics Error:", e);
    }
}

// =====================================================
// UTILS
// =====================================================

function getCategoryColor(cat) {
    const colors = {
        'Plastic': '#ff4444',
        'Metal': '#00d4ff',
        'Glass': '#ffcc00',
        'Paper': '#ffffff',
        'Organic': '#00ff88',
        'Unknown': '#a0a0a0'
    };
    return colors[cat] || '#fff';
}
