/**
 * BrewLab — Frontend Controller
 * Handles slider binding, API calls to /predict & /recommend,
 * gauge animation, and dynamic "Fix My Brew" recourse UI.
 */

const API_BASE = window.location.origin;

// ── State ──────────────────────────────────────────────────
let selectedRoast = 'Medium';
let lastPrediction = null;

// ── DOM References ─────────────────────────────────────────
const DOM = {
    // Sliders
    grind:      document.getElementById('grind_size'),
    temp:       document.getElementById('water_temp'),
    time:       document.getElementById('brew_time'),
    ratio:      document.getElementById('water_ratio'),
    // Value displays
    valGrind:   document.getElementById('val-grind'),
    valTemp:    document.getElementById('val-temp'),
    valTime:    document.getElementById('val-time'),
    valRatio:   document.getElementById('val-ratio'),
    // Roast selector
    roastBtns:  document.querySelectorAll('.roast-btn'),
    // Predict button
    predictBtn:     document.getElementById('predict-btn'),
    predictText:    document.getElementById('predict-btn-text'),
    predictLoading: document.getElementById('predict-btn-loading'),
    // Gauge
    gaugePointer:   document.getElementById('gauge-pointer'),
    // Result
    resultPlaceholder: document.getElementById('result-placeholder'),
    resultDisplay:     document.getElementById('result-display'),
    resultIcon:        document.getElementById('result-icon'),
    resultLabel:       document.getElementById('result-label'),
    resultConfidence:  document.getElementById('result-confidence'),
    confidenceBars:    document.getElementById('confidence-bars'),
    // Fix section
    fixSection:     document.getElementById('fix-section'),
    fixBtn:         document.getElementById('fix-btn'),
    fixBtnText:     document.getElementById('fix-btn-text'),
    fixBtnLoading:  document.getElementById('fix-btn-loading'),
    fixResults:     document.getElementById('fix-results'),
    mutableFeatures: document.getElementById('mutable-features'),
};


// ── Slider Live Updates ────────────────────────────────────
function bindSlider(slider, display, formatter) {
    const update = () => { display.textContent = formatter(slider.value); };
    slider.addEventListener('input', update);
    update();
}

bindSlider(DOM.grind, DOM.valGrind, v => parseInt(v).toString());
bindSlider(DOM.temp,  DOM.valTemp,  v => parseFloat(v).toFixed(1));
bindSlider(DOM.time,  DOM.valTime,  v => parseInt(v).toString());
bindSlider(DOM.ratio, DOM.valRatio, v => parseFloat(v).toFixed(1));


// ── Roast Selector ─────────────────────────────────────────
DOM.roastBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        selectedRoast = btn.dataset.roast;
        DOM.roastBtns.forEach(b => {
            const isActive = b.dataset.roast === selectedRoast;
            b.classList.toggle('border-espresso-500', isActive);
            b.classList.toggle('bg-espresso-950/60', isActive);
            b.classList.toggle('ring-1', isActive);
            b.classList.toggle('ring-espresso-500/30', isActive);
            b.classList.toggle('border-surface-line', !isActive);
            b.classList.toggle('bg-surface-base/50', !isActive);
            // Update label text color
            const label = b.querySelector('span:last-child');
            label.classList.toggle('text-espresso-100', isActive);
            label.classList.toggle('text-espresso-300', !isActive);
        });
    });
});


// ── Build Brew Payload ─────────────────────────────────────
function getBrewPayload() {
    return {
        roast_level:        selectedRoast,
        grind_size_microns: parseFloat(DOM.grind.value),
        water_temp_c:       parseFloat(DOM.temp.value),
        brew_time_seconds:  parseFloat(DOM.time.value),
        water_ratio:        parseFloat(DOM.ratio.value),
    };
}


// ── Predict ────────────────────────────────────────────────
DOM.predictBtn.addEventListener('click', async () => {
    const payload = getBrewPayload();

    // Loading state
    setButtonLoading(DOM.predictBtn, DOM.predictText, DOM.predictLoading, true);

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error(`Server returned ${res.status}`);

        const data = await res.json();
        lastPrediction = data;
        renderPrediction(data);

    } catch (err) {
        console.error('Prediction error:', err);
        showError('Could not reach the backend. Is the FastAPI server running on port 8000?');
    } finally {
        setButtonLoading(DOM.predictBtn, DOM.predictText, DOM.predictLoading, false);
    }
});


// ── Render Prediction ──────────────────────────────────────
function renderPrediction(data) {
    const { prediction, confidence_scores } = data;

    // Hide placeholder, show result
    DOM.resultPlaceholder.classList.add('hidden');
    DOM.resultDisplay.classList.remove('hidden');
    DOM.resultDisplay.classList.add('animate-fade-up');

    // Determine category
    const isSour     = prediction.toLowerCase().includes('sour');
    const isBitter   = prediction.toLowerCase().includes('bitter');
    const isBalanced  = prediction.toLowerCase().includes('balanced');

    // Icon and label
    if (isSour) {
        DOM.resultIcon.textContent = '🍋';
        DOM.resultIcon.className = 'w-16 h-16 mx-auto mb-3 rounded-full flex items-center justify-center text-3xl status-sour border';
        DOM.resultLabel.textContent = 'Under-Extracted';
        DOM.resultLabel.className = 'font-display text-2xl font-bold mb-1 text-yellow-400';
    } else if (isBitter) {
        DOM.resultIcon.textContent = '🔥';
        DOM.resultIcon.className = 'w-16 h-16 mx-auto mb-3 rounded-full flex items-center justify-center text-3xl status-bitter border';
        DOM.resultLabel.textContent = 'Over-Extracted';
        DOM.resultLabel.className = 'font-display text-2xl font-bold mb-1 text-espresso-400';
    } else {
        DOM.resultIcon.textContent = '✨';
        DOM.resultIcon.className = 'w-16 h-16 mx-auto mb-3 rounded-full flex items-center justify-center text-3xl status-balanced border';
        DOM.resultLabel.textContent = 'Balanced';
        DOM.resultLabel.className = 'font-display text-2xl font-bold mb-1 text-green-400';
    }

    // Confidence text
    const topScore = Math.max(...Object.values(confidence_scores));
    DOM.resultConfidence.textContent = `${(topScore * 100).toFixed(1)}% confidence`;

    // Move gauge pointer
    animateGauge(confidence_scores);

    // Render confidence bars
    renderConfidenceBars(confidence_scores);

    // Show or hide Fix section
    if (isSour || isBitter) {
        DOM.fixSection.classList.remove('hidden');
        DOM.fixResults.classList.add('hidden');
        DOM.fixResults.innerHTML = '';
    } else {
        DOM.fixSection.classList.add('hidden');
    }
}


// ── Gauge Animation ────────────────────────────────────────
function animateGauge(scores) {
    // Map scores to a 0-100 position on the spectrum:
    //   0 = far left (Sour), 50 = center (Balanced), 100 = far right (Bitter)
    const sourKey    = Object.keys(scores).find(k => k.toLowerCase().includes('sour'))     || '';
    const balancedKey = Object.keys(scores).find(k => k.toLowerCase().includes('balanced')) || '';
    const bitterKey  = Object.keys(scores).find(k => k.toLowerCase().includes('bitter'))   || '';

    const sourScore    = scores[sourKey]    || 0;
    const balancedScore = scores[balancedKey] || 0;
    const bitterScore  = scores[bitterKey]  || 0;

    // Weighted position: sour pulls left, bitter pulls right
    const position = (sourScore * 10) + (balancedScore * 50) + (bitterScore * 90);

    DOM.gaugePointer.style.opacity = '1';
    DOM.gaugePointer.style.left = `${position}%`;
}


// ── Confidence Bars ────────────────────────────────────────
function renderConfidenceBars(scores) {
    const configs = [
        { key: 'Sour',     color: '#e5b83a', label: 'Sour / Under' },
        { key: 'Balanced', color: '#4ade80', label: 'Balanced' },
        { key: 'Bitter',   color: '#a36e38', label: 'Bitter / Over' },
    ];

    DOM.confidenceBars.innerHTML = configs.map(cfg => {
        const scoreKey = Object.keys(scores).find(k => k.toLowerCase().includes(cfg.key.toLowerCase())) || '';
        const value = scores[scoreKey] || 0;
        const pct = (value * 100).toFixed(1);

        return `
            <div>
                <div class="flex justify-between items-center mb-1">
                    <span class="text-xs font-medium text-espresso-300">${cfg.label}</span>
                    <span class="text-xs font-semibold tabular-nums" style="color: ${cfg.color}">${pct}%</span>
                </div>
                <div class="confidence-bar-track">
                    <div class="confidence-bar-fill" style="width: ${pct}%; background: ${cfg.color};"></div>
                </div>
            </div>
        `;
    }).join('');
}


// ── Fix My Brew ────────────────────────────────────────────
DOM.fixBtn.addEventListener('click', async () => {
    const checked = [...DOM.mutableFeatures.querySelectorAll('input:checked')].map(cb => cb.value);

    if (checked.length === 0) {
        DOM.fixResults.classList.remove('hidden');
        DOM.fixResults.innerHTML = `
            <div class="fix-fail text-center">
                <p class="text-sm text-red-400 font-medium">Select at least one parameter you can adjust.</p>
            </div>
        `;
        return;
    }

    const payload = {
        cup: getBrewPayload(),
        mutable_features: checked,
        target_class: 'Balanced',
        confidence: 0.60,
    };

    setButtonLoading(DOM.fixBtn, DOM.fixBtnText, DOM.fixBtnLoading, true);

    try {
        const res = await fetch(`${API_BASE}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error(`Server returned ${res.status}`);

        const data = await res.json();
        renderRecourse(data);

    } catch (err) {
        console.error('Recourse error:', err);
        DOM.fixResults.classList.remove('hidden');
        DOM.fixResults.innerHTML = `
            <div class="fix-fail text-center">
                <p class="text-sm text-red-400 font-medium">Failed to calculate fix. Check backend console.</p>
            </div>
        `;
    } finally {
        setButtonLoading(DOM.fixBtn, DOM.fixBtnText, DOM.fixBtnLoading, false);
    }
});


// ── Render Recourse ────────────────────────────────────────
const FEATURE_META = {
    grind_size_microns:  { label: 'Grind Size',   unit: 'µm',  icon: '⚙️' },
    water_temp_c:        { label: 'Water Temp',   unit: '°C',  icon: '🌡️' },
    brew_time_seconds:   { label: 'Brew Time',    unit: 'sec',  icon: '⏱️' },
    water_ratio:         { label: 'Water Ratio',  unit: ':1',   icon: '💧' },
};

function renderRecourse(data) {
    DOM.fixResults.classList.remove('hidden');
    DOM.fixResults.classList.add('reveal-enter');

    const achieved = data.confidence_achieved;
    const changes  = data.recommended_changes || {};

    if (!achieved) {
        DOM.fixResults.innerHTML = `
            <div class="fix-fail text-center">
                <p class="text-sm text-red-400 font-semibold mb-1">Infeasible Target</p>
                <p class="text-xs text-espresso-500">The AI couldn't reach 60% confidence for "Balanced" with only the selected parameters. Try unlocking more variables.</p>
            </div>
        `;
        return;
    }

    const changeKeys = Object.keys(changes);
    if (changeKeys.length === 0) {
        DOM.fixResults.innerHTML = `
            <div class="fix-success text-center">
                <p class="text-sm text-green-400 font-semibold">No changes needed — your brew is already on target!</p>
            </div>
        `;
        return;
    }

    let html = `
        <div class="fix-success mb-4 flex items-center gap-2">
            <svg class="w-5 h-5 text-green-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
            <p class="text-sm text-green-400 font-medium">Optimal fix found. Adjust these parameters:</p>
        </div>
        <div class="space-y-2.5">
    `;

    changeKeys.forEach((key, i) => {
        const change = changes[key];
        const meta   = FEATURE_META[key] || { label: key, unit: '', icon: '📐' };
        const from   = parseFloat(change.from);
        const to     = parseFloat(change.to);
        const delta  = to - from;
        const isUp   = delta > 0;

        const delayClass = `animation-delay: ${i * 80}ms;`;

        html += `
            <div class="delta-card reveal-enter" style="${delayClass}">
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-3">
                        <span class="text-xl">${meta.icon}</span>
                        <div>
                            <p class="text-sm font-semibold text-espresso-200">${meta.label}</p>
                            <p class="text-xs text-espresso-500 mt-0.5">
                                <span class="tabular-nums">${formatVal(from, key)}</span>
                                <span class="mx-1.5 text-espresso-700">→</span>
                                <span class="tabular-nums font-semibold text-espresso-300">${formatVal(to, key)}</span>
                                <span class="text-espresso-600">${meta.unit}</span>
                            </p>
                        </div>
                    </div>
                    <div class="flex items-center gap-2">
                        <div class="delta-arrow ${isUp ? 'increase' : 'decrease'}">
                            ${isUp ? '↑' : '↓'}
                        </div>
                        <span class="text-sm font-bold tabular-nums ${isUp ? 'text-green-400' : 'text-red-400'}">
                            ${isUp ? '+' : ''}${formatVal(delta, key)}
                        </span>
                    </div>
                </div>
            </div>
        `;
    });

    html += '</div>';
    DOM.fixResults.innerHTML = html;
}


// ── Helpers ────────────────────────────────────────────────
function formatVal(val, key) {
    if (key === 'water_ratio') return val.toFixed(1);
    if (key === 'water_temp_c') return val.toFixed(1);
    return Math.round(val).toString();
}

function setButtonLoading(btn, textEl, loadingEl, isLoading) {
    btn.disabled = isLoading;
    textEl.classList.toggle('hidden', isLoading);
    loadingEl.classList.toggle('hidden', !isLoading);
    if (isLoading) {
        loadingEl.classList.add('flex');
    } else {
        loadingEl.classList.remove('flex');
    }
    btn.classList.toggle('opacity-70', isLoading);
    btn.classList.toggle('cursor-wait', isLoading);
}

function showError(message) {
    DOM.resultPlaceholder.classList.add('hidden');
    DOM.resultDisplay.classList.remove('hidden');
    DOM.resultIcon.textContent = '⚠️';
    DOM.resultIcon.className = 'w-16 h-16 mx-auto mb-3 rounded-full flex items-center justify-center text-3xl bg-red-950/30 border border-red-800/30';
    DOM.resultLabel.textContent = 'Connection Error';
    DOM.resultLabel.className = 'font-display text-2xl font-bold mb-1 text-red-400';
    DOM.resultConfidence.textContent = message;
    DOM.confidenceBars.innerHTML = '';
    DOM.gaugePointer.style.opacity = '0';
    DOM.fixSection.classList.add('hidden');
}
