/**
 * NASDAQ AI Tahmin Merkezi — Multi-Model Frontend Logic
 * Handles ticker list, search/filter, multi-model predictions (XGBoost, LightGBM, Random Forest, SVM),
 * consensus banners, and background task polling.
 */

// ─── State ──────────────────────────────────────────────────
let allTickers = [];
let filteredTickers = [];
let selectedTicker = null;
let pollingInterval = null;
let currentTaskResult = null;
let activeFiModel = 'xgboost';

// ─── DOM Elements ───────────────────────────────────────────
const searchInput = document.getElementById('search-input');
const searchClear = document.getElementById('search-clear');
const sectorFilter = document.getElementById('sector-filter');
const stockList = document.getElementById('stock-list');
const listLoading = document.getElementById('list-loading');
const actionBar = document.getElementById('action-bar');
const selectedSymbolEl = document.getElementById('selected-symbol');
const selectedNameEl = document.getElementById('selected-name');
const analyzeBtn = document.getElementById('analyze-btn');
const filteredCountEl = document.getElementById('filtered-count');
const statTotalTickers = document.getElementById('stat-total-tickers');

// Results Header & Panels
const emptyState = document.getElementById('empty-state');
const resultsContent = document.getElementById('results-content');
const resultTickerSymbol = document.getElementById('result-ticker-symbol');
const resultTickerName = document.getElementById('result-ticker-name');
const resultDate = document.getElementById('result-date');

// Universal Section
const universalConsensusBanner = document.getElementById('universal-consensus-banner');
const universalConsensusTitle = document.getElementById('universal-consensus-title');
const universalConsensusDesc = document.getElementById('universal-consensus-desc');
const universalConsensusIcon = document.getElementById('universal-consensus-icon');
const universalModelsGrid = document.getElementById('universal-models-grid');

// Specific Section
const cardSpecific = document.getElementById('card-specific');
const specificBadge = document.getElementById('specific-badge');
const trainingProgress = document.getElementById('training-progress');
const trainingResult = document.getElementById('training-result');
const trainingError = document.getElementById('training-error');
const progressBar = document.getElementById('progress-bar');
const progressPct = document.getElementById('progress-pct');
const progressStage = document.getElementById('progress-stage');

const specificConsensusBanner = document.getElementById('specific-consensus-banner');
const specificConsensusTitle = document.getElementById('specific-consensus-title');
const specificConsensusDesc = document.getElementById('specific-consensus-desc');
const specificConsensusIcon = document.getElementById('specific-consensus-icon');
const specificModelsGrid = document.getElementById('specific-models-grid');

const fiModelTabs = document.getElementById('fi-model-tabs');
const fiBars = document.getElementById('fi-bars');
const errorMessage = document.getElementById('error-message');

// Price Card
const cardPrice = document.getElementById('card-price');
const priceCurrent = document.getElementById('price-current');
const priceCandleTime = document.getElementById('price-candle-time');
const priceOpen = document.getElementById('price-open');
const priceHigh = document.getElementById('price-high');
const priceLow = document.getElementById('price-low');
const priceVolume = document.getElementById('price-volume');

// Explanation & Feature Gauges
const predictionExplanation = document.getElementById('prediction-explanation');
const featureGauges = document.getElementById('feature-gauges');
const gaugeGrid = document.getElementById('gauge-grid');


// ─── Initialize ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadTickers();
    searchInput.addEventListener('input', handleSearch);
    searchClear.addEventListener('click', clearSearch);
    sectorFilter.addEventListener('change', handleSearch);
    analyzeBtn.addEventListener('click', runAnalysis);

    // Setup FI model tab switching
    if (fiModelTabs) {
        fiModelTabs.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                fiModelTabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                activeFiModel = btn.dataset.model;
                if (currentTaskResult) {
                    renderFeatureImportance(currentTaskResult);
                }
            });
        });
    }
});


// ─── Load Tickers ───────────────────────────────────────────
async function loadTickers() {
    try {
        const resp = await fetch('/api/tickers');
        const data = await resp.json();

        if (data.status !== 'ok') throw new Error(data.message || 'Failed to load tickers');

        allTickers = data.tickers;
        filteredTickers = [...allTickers];

        statTotalTickers.querySelector('.stat-value').textContent = allTickers.length.toLocaleString();

        const sectors = [...new Set(allTickers.map(t => t.sector).filter(s => s && s !== '' && s !== 'None'))].sort();
        sectors.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s;
            opt.textContent = s;
            sectorFilter.appendChild(opt);
        });

        renderStockList();
        listLoading.classList.add('hidden');
    } catch (err) {
        listLoading.innerHTML = `<span style="color: var(--sell);">❌ Hata: ${err.message}</span>`;
    }
}


// ─── Search & Filter ────────────────────────────────────────
function handleSearch() {
    const query = searchInput.value.trim().toLowerCase();
    const sector = sectorFilter.value;

    searchClear.style.display = query ? 'block' : 'none';

    filteredTickers = allTickers.filter(t => {
        const matchesQuery = !query ||
            t.symbol.toLowerCase().includes(query) ||
            t.name.toLowerCase().includes(query);
        const matchesSector = !sector || t.sector === sector;
        return matchesQuery && matchesSector;
    });

    renderStockList();
}

function clearSearch() {
    searchInput.value = '';
    searchClear.style.display = 'none';
    sectorFilter.value = '';
    filteredTickers = [...allTickers];
    renderStockList();
    searchInput.focus();
}


// ─── Render Stock List ──────────────────────────────────────
function renderStockList() {
    filteredCountEl.textContent = `${filteredTickers.length} sonuç`;

    const RENDER_LIMIT = 50;
    const toRender = filteredTickers.slice(0, RENDER_LIMIT);

    stockList.innerHTML = '';

    if (toRender.length === 0) {
        stockList.innerHTML = `
            <div style="text-align:center; padding:40px 20px; color: var(--text-muted);">
                <div style="font-size:2rem; margin-bottom:8px;">🔍</div>
                <p>Sonuç bulunamadı</p>
            </div>
        `;
        return;
    }

    const fragment = document.createDocumentFragment();

    toRender.forEach(ticker => {
        const item = document.createElement('div');
        item.className = 'stock-item';
        if (selectedTicker && selectedTicker.symbol === ticker.symbol) {
            item.classList.add('selected');
        }
        item.dataset.symbol = ticker.symbol;

        item.innerHTML = `
            <span class="stock-symbol">${ticker.symbol}</span>
            <span class="stock-name" title="${escapeHtml(ticker.name)}">${escapeHtml(ticker.name)}</span>
            <span class="stock-price">${ticker.lastsale || ''}</span>
        `;

        item.addEventListener('click', () => selectTicker(ticker));
        fragment.appendChild(item);
    });

    if (filteredTickers.length > RENDER_LIMIT) {
        const more = document.createElement('div');
        more.style.cssText = 'text-align:center; padding:12px; color: var(--text-muted); font-size:0.75rem;';
        more.textContent = `... ve ${filteredTickers.length - RENDER_LIMIT} hisse daha (aramayı daraltın)`;
        fragment.appendChild(more);
    }

    stockList.appendChild(fragment);
}


// ─── Select Ticker ──────────────────────────────────────────
function selectTicker(ticker) {
    selectedTicker = ticker;

    document.querySelectorAll('.stock-item').forEach(item => {
        item.classList.toggle('selected', item.dataset.symbol === ticker.symbol);
    });

    actionBar.style.display = 'flex';
    selectedSymbolEl.textContent = ticker.symbol;
    selectedNameEl.textContent = ticker.name;
    analyzeBtn.disabled = false;
}


// ─── Run Analysis ───────────────────────────────────────────
async function runAnalysis() {
    if (!selectedTicker) return;

    const ticker = selectedTicker;
    currentTaskResult = null;

    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }

    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;

    emptyState.style.display = 'none';
    resultsContent.style.display = 'flex';

    resultTickerSymbol.textContent = ticker.symbol;
    resultTickerName.textContent = ticker.name;

    resetResultCards();

    try {
        const resp = await fetch(`/api/predict/${ticker.symbol}`, { method: 'POST' });
        const data = await resp.json();

        if (data.status !== 'ok') throw new Error(data.message || 'Prediction failed');

        // ── Price Info ──
        if (data.price_info) {
            const p = data.price_info;
            cardPrice.style.display = 'block';
            priceCurrent.textContent = '$' + p.current_price.toLocaleString('en-US', {minimumFractionDigits: 2});
            priceCandleTime.textContent = 'Son mum: ' + p.last_candle_time;
            priceOpen.textContent = '$' + p.open.toFixed(2);
            priceHigh.textContent = '$' + p.high.toFixed(2);
            priceLow.textContent = '$' + p.low.toFixed(2);
            priceVolume.textContent = p.volume.toLocaleString();
        }

        // ── Universal Results ──
        if (data.universal && !data.universal.error) {
            renderUniversalResults(data.universal, ticker.symbol, data.price_info);
        } else {
            universalConsensusBanner.style.display = 'none';
            universalModelsGrid.innerHTML = `<div class="error-msg">⚠️ ${data.universal?.error || 'Universal modeller bulunamadı'}</div>`;
        }

        // ── Background task polling ──
        if (data.task_id) {
            pollTaskStatus(data.task_id);
        }

    } catch (err) {
        universalModelsGrid.innerHTML = `<div class="error-msg">❌ Hata: ${err.message}</div>`;
    } finally {
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;
    }
}


// ─── Render Universal Models ────────────────────────────────
function renderUniversalResults(universalData, symbol, priceInfo) {
    const lastDate = universalData.last_date || '';
    resultDate.textContent = lastDate;

    // Consensus Banner
    if (universalData.consensus) {
        const c = universalData.consensus;
        const isBuy = c.signal === 'BUY';
        universalConsensusBanner.style.display = 'flex';
        universalConsensusBanner.className = `consensus-banner ${isBuy ? 'buy' : 'sell'}`;
        universalConsensusIcon.textContent = isBuy ? '🚀' : '📉';
        universalConsensusTitle.textContent = `Universal Konsensüs: ${isBuy ? 'AL (BUY)' : 'SAT (SELL)'}`;
        universalConsensusDesc.textContent = `${c.buy_count} / ${c.total_models} Model AL Yönünde · Ort. Güven: %${c.avg_confidence}`;

        const priceStr = priceInfo ? ` $${priceInfo.current_price.toFixed(2)}` : '';
        const signalTR = isBuy ? 'YÜKSELECEĞİNİ' : 'DÜŞECEĞİNİ';
        predictionExplanation.innerHTML = `
            <strong>Ne anlama geliyor?</strong> Universal modeller, <strong>${symbol}</strong> hissesinin
            son mum verilerine (${lastDate}) dayanarak, bir sonraki mumda fiyatın${priceStr} seviyesinden
            <strong>${signalTR}</strong> tahmin ediyor.
            <br><em>Aşağıda XGBoost, LightGBM ve Random Forest modellerinin bağımsız tahminleri listelenmiştir.</em>
        `;
    }

    // Models Grid
    universalModelsGrid.innerHTML = '';
    if (universalData.models) {
        for (const [key, m] of Object.entries(universalData.models)) {
            if (m.error) {
                const card = document.createElement('div');
                card.className = 'model-card card-error';
                card.innerHTML = `
                    <div class="model-card-header">
                        <span class="model-icon">${m.icon || '🤖'}</span>
                        <span class="model-name">${m.name}</span>
                    </div>
                    <p class="model-error-text">Henüz eğitilmedi</p>
                `;
                universalModelsGrid.appendChild(card);
                continue;
            }

            const isBuy = m.signal === 'BUY';
            const card = document.createElement('div');
            card.className = `model-card ${isBuy ? 'buy' : 'sell'}`;
            card.innerHTML = `
                <div class="model-card-header">
                    <div class="model-title-group">
                        <span class="model-icon">${m.icon}</span>
                        <span class="model-name">${m.name}</span>
                    </div>
                    <span class="model-file-tag">${m.model_file || ''}</span>
                </div>
                <div class="model-card-body">
                    <div class="model-signal-badge ${isBuy ? 'buy' : 'sell'}">
                        <span class="signal-arrow">${isBuy ? '▲' : '▼'}</span>
                        <span class="signal-name">${isBuy ? 'BUY — AL' : 'SELL — SAT'}</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-label">Güven Oranı</span>
                        <span class="metric-val">%${m.confidence}</span>
                    </div>
                </div>
            `;
            universalModelsGrid.appendChild(card);
        }
    }

    // Feature gauges
    if (universalData.features) {
        featureGauges.style.display = 'block';
        renderFeatureGauges(universalData.features);
    }
}


// ─── Reset Result Cards ─────────────────────────────────────
function resetResultCards() {
    cardPrice.style.display = 'none';
    priceCurrent.textContent = '—';
    priceOpen.textContent = '—';
    priceHigh.textContent = '—';
    priceLow.textContent = '—';
    priceVolume.textContent = '—';
    priceCandleTime.textContent = '';

    universalConsensusBanner.style.display = 'none';
    universalModelsGrid.innerHTML = '<div class="spinner-container"><div class="spinner"></div><span>Universal modeller hesaplanıyor...</span></div>';
    resultDate.textContent = '';
    predictionExplanation.innerHTML = '';
    featureGauges.style.display = 'none';
    gaugeGrid.innerHTML = '';

    specificBadge.textContent = '🔄 Eğitiliyor';
    specificBadge.className = 'card-badge badge-training';
    trainingProgress.style.display = 'block';
    trainingResult.style.display = 'none';
    trainingError.style.display = 'none';
    progressBar.style.width = '0%';
    progressPct.textContent = '0%';
    progressStage.textContent = 'Başlatılıyor...';
    cardSpecific.classList.remove('completed');
    fiBars.innerHTML = '';
}


// ─── Poll Task Status ───────────────────────────────────────
function pollTaskStatus(taskId) {
    pollingInterval = setInterval(async () => {
        try {
            const resp = await fetch(`/api/status/${taskId}`);
            const task = await resp.json();

            progressBar.style.width = task.progress + '%';
            progressPct.textContent = task.progress + '%';
            progressStage.textContent = task.message || '';

            if (task.status === 'done') {
                clearInterval(pollingInterval);
                pollingInterval = null;
                currentTaskResult = task.result;
                showSpecificResult(task.result);
            } else if (task.status === 'error') {
                clearInterval(pollingInterval);
                pollingInterval = null;
                showSpecificError(task.message);
            }
        } catch (err) {
            // Retry silently
        }
    }, 1200);
}


// ─── Show Specific Results ──────────────────────────────────
function showSpecificResult(result) {
    specificBadge.textContent = '✅ Tamamlandı';
    specificBadge.className = 'card-badge badge-done';
    trainingProgress.style.display = 'none';
    trainingResult.style.display = 'block';
    trainingError.style.display = 'none';

    // Stock-Specific Consensus Banner
    if (result.consensus) {
        const c = result.consensus;
        const isBuy = c.signal === 'BUY';
        specificConsensusBanner.className = `consensus-banner ${isBuy ? 'buy' : 'sell'}`;
        specificConsensusIcon.textContent = isBuy ? '🏆' : '⚠️';
        specificConsensusTitle.textContent = `Hisseye Özel Consensus: ${isBuy ? 'AL (BUY)' : 'SAT (SELL)'}`;
        specificConsensusDesc.textContent = `${result.ticker} için ${c.buy_count} / 3 Model AL Sinyali Veriyor · Ort. Güven: %${c.avg_confidence} · Toplam Veri: ${result.total_records?.toLocaleString() || '—'} Mum`;
    }

    // 4 Stock-Specific Model Cards Grid
    specificModelsGrid.innerHTML = '';
    if (result.models) {
        for (const [key, m] of Object.entries(result.models)) {
            const isBuy = m.signal === 'BUY';
            const card = document.createElement('div');
            card.className = `model-card ${isBuy ? 'buy' : 'sell'}`;
            card.innerHTML = `
                <div class="model-card-header">
                    <div class="model-title-group">
                        <span class="model-icon">${m.icon}</span>
                        <span class="model-name">${m.name}</span>
                    </div>
                    <span class="model-file-tag">${m.model_file}</span>
                </div>
                <div class="model-card-body">
                    <div class="model-signal-badge ${isBuy ? 'buy' : 'sell'}">
                        <span class="signal-arrow">${isBuy ? '▲' : '▼'}</span>
                        <span class="signal-name">${isBuy ? 'BUY — AL' : 'SELL — SAT'}</span>
                    </div>
                    <div class="model-metrics-grid">
                        <div class="model-metric">
                            <span class="metric-label">Güven</span>
                            <span class="metric-val">%${m.confidence}</span>
                        </div>
                        <div class="model-metric">
                            <span class="metric-label">Accuracy</span>
                            <span class="metric-val">%${(m.accuracy * 100).toFixed(1)}</span>
                        </div>
                        <div class="model-metric">
                            <span class="metric-label">Buy Precision</span>
                            <span class="metric-val">%${(m.precision_buy * 100).toFixed(1)}</span>
                        </div>
                    </div>
                </div>
            `;
            specificModelsGrid.appendChild(card);
        }
    }

    // Render Feature Importance for active tab
    renderFeatureImportance(result);

    cardSpecific.classList.add('completed');
}


// ─── Render Feature Importance Tabs ──────────────────────────
function renderFeatureImportance(result) {
    if (!result || !result.models) return;
    const modelData = result.models[activeFiModel];
    if (!modelData || !modelData.feature_importance || modelData.feature_importance.length === 0) {
        fiBars.innerHTML = `<div class="fi-empty">Bu model için (ör. SVM) feature importance doğrudan mevcut değildir.</div>`;
        return;
    }

    const fi = modelData.feature_importance;
    const maxImp = Math.max(...fi.map(f => f.importance));

    fiBars.innerHTML = fi.map(f => `
        <div class="fi-bar-item">
            <span class="fi-bar-label">${f.feature}</span>
            <div class="fi-bar-track">
                <div class="fi-bar-fill" style="width: ${maxImp > 0 ? (f.importance / maxImp * 100).toFixed(1) : 0}%"></div>
            </div>
            <span class="fi-bar-value">${(f.importance * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}


// ─── Show Specific Error ────────────────────────────────────
function showSpecificError(message) {
    specificBadge.textContent = '❌ Hata';
    specificBadge.className = 'card-badge badge-error';
    trainingProgress.style.display = 'none';
    trainingResult.style.display = 'none';
    trainingError.style.display = 'flex';
    errorMessage.textContent = message;
}


// ─── Utility ────────────────────────────────────────────────
function escapeHtml(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}


// ─── Feature Gauges Renderer ────────────────────────────────
const featureDescriptions = {
    'RSI_14': {
        label: 'RSI (14)',
        desc: (v) => {
            if (v > 70) return 'Aşırı alım bölgesi — düşüş sinyali';
            if (v < 30) return 'Aşırı satım bölgesi — yükseliş sinyali';
            return 'Nötr bölge';
        },
        cssClass: (v) => v > 70 ? 'rsi-overbought' : v < 30 ? 'rsi-oversold' : 'rsi-neutral',
        format: (v) => v.toFixed(1),
    },
    'MACD_Hist': {
        label: 'MACD Histogram',
        desc: (v) => v > 0 ? 'Pozitif — yükseliş momentumu' : 'Negatif — düşüş momentumu',
        cssClass: (v) => v > 0 ? 'macd-positive' : 'macd-negative',
        format: (v) => v.toFixed(4),
    },
    'SMA_20_Ratio': {
        label: 'Fiyat / SMA(20)',
        desc: (v) => v > 1 ? `Fiyat 20 saatlik ortalamanın %${((v-1)*100).toFixed(1)} üstünde` : `Fiyat 20 saatlik ortalamanın %${((1-v)*100).toFixed(1)} altında`,
        format: (v) => v.toFixed(4),
    },
    'SMA_50_Ratio': {
        label: 'Fiyat / SMA(50)',
        desc: (v) => v > 1 ? `Fiyat 50 saatlik ortalamanın %${((v-1)*100).toFixed(1)} üstünde` : `Fiyat 50 saatlik ortalamanın %${((1-v)*100).toFixed(1)} altında`,
        format: (v) => v.toFixed(4),
    },
    'Hourly_Return': {
        label: 'Son Saatlik Getiri',
        desc: (v) => `Son 1 saatte %${(v*100).toFixed(3)} ${v >= 0 ? 'yükseliş' : 'düşüş'}`,
        format: (v) => (v * 100).toFixed(3) + '%',
    },
    'Return_lag_1': {
        label: '1 Saat Önceki Getiri',
        desc: () => 'Bir önceki saatlik değişim',
        format: (v) => (v * 100).toFixed(3) + '%',
    },
    'Return_lag_2': {
        label: '2 Saat Önceki Getiri',
        desc: () => 'İki saat önceki değişim',
        format: (v) => (v * 100).toFixed(3) + '%',
    },
    'Return_lag_3': {
        label: '3 Saat Önceki Getiri',
        desc: () => 'Üç saat önceki değişim',
        format: (v) => (v * 100).toFixed(3) + '%',
    },
};

function renderFeatureGauges(features) {
    gaugeGrid.innerHTML = '';

    for (const [key, value] of Object.entries(features)) {
        const config = featureDescriptions[key] || {
            label: key,
            desc: () => '',
            format: (v) => v.toFixed(4),
        };

        const cssClass = config.cssClass ? config.cssClass(value) : '';

        const item = document.createElement('div');
        item.className = `gauge-item ${cssClass}`;
        item.innerHTML = `
            <span class="gauge-label">${config.label}</span>
            <span class="gauge-value">${config.format(value)}</span>
            <span class="gauge-desc">${config.desc(value)}</span>
        `;
        gaugeGrid.appendChild(item);
    }
}

