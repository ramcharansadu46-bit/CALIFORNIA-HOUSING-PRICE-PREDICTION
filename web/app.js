/**
 * California Housing Predictive Analysis - Interactive Frontend Logic
 */

// Preset Districts Data
const PRESETS = {
  sf_bay: {
    longitude: -122.25,
    latitude: 37.87,
    housing_median_age: 42,
    total_rooms: 1650,
    total_bedrooms: 280,
    population: 680,
    households: 270,
    median_income: 8.5,
    ocean_proximity: 'NEAR BAY'
  },
  silicon_valley: {
    longitude: -121.90,
    latitude: 37.35,
    housing_median_age: 20,
    total_rooms: 4200,
    total_bedrooms: 680,
    population: 1750,
    households: 650,
    median_income: 9.2,
    ocean_proximity: '<1H OCEAN'
  },
  la_coastal: {
    longitude: -118.39,
    latitude: 34.12,
    housing_median_age: 29,
    total_rooms: 6447,
    total_bedrooms: 1012,
    population: 2184,
    households: 960,
    median_income: 8.28,
    ocean_proximity: '<1H OCEAN'
  },
  san_diego: {
    longitude: -117.16,
    latitude: 32.72,
    housing_median_age: 34,
    total_rooms: 3100,
    total_bedrooms: 540,
    population: 1400,
    households: 520,
    median_income: 5.4,
    ocean_proximity: 'NEAR OCEAN'
  },
  inland_valley: {
    longitude: -119.78,
    latitude: 36.74,
    housing_median_age: 32,
    total_rooms: 1850,
    total_bedrooms: 390,
    population: 1250,
    households: 370,
    median_income: 2.35,
    ocean_proximity: 'INLAND'
  },
  sacramento: {
    longitude: -121.49,
    latitude: 38.58,
    housing_median_age: 26,
    total_rooms: 2800,
    total_bedrooms: 480,
    population: 1350,
    households: 470,
    median_income: 4.15,
    ocean_proximity: 'INLAND'
  }
};

// 20 Figures Metadata Catalog
const FIGURES = [
  { id: 'fig01', file: 'fig01_data_quality.png', title: 'Data Quality & Missingness Audit', desc: 'Missing values audit showing exactly 207 rows missing total_bedrooms (1.00%), zero duplicate rows, and data types.' },
  { id: 'fig02', file: 'fig02_distributions.png', title: 'Feature Distributions & Skewness', desc: 'Histograms and density plots for all raw census features, exhibiting right skewness in room counts and income.' },
  { id: 'fig03', file: 'fig03_correlation.png', title: 'Pearson Correlation Matrix', desc: 'Pairwise feature correlations revealing median_income as dominant price driver (r = 0.69).' },
  { id: 'fig04', file: 'fig04_geography.png', title: 'Geographic Price Heatmap', desc: 'Spatial coordinate scatter plot colored by house value with population density sizing, showing extreme coastal premium.' },
  { id: 'fig05', file: 'fig05_imputation.png', title: 'KNN vs Median Imputation', desc: 'Distribution preservation of imputed bedrooms; KNN preserves bivariate room-bedroom ratios without test leakage.' },
  { id: 'fig06', file: 'fig06_income_relationship.png', title: 'Non-Linear Income Relationship', desc: 'Binned scatter plot demonstrating non-linear scaling between income and price, plus right-censored ceiling at $500,001.' },
  { id: 'fig07', file: 'fig07_income_deciles_residuals.png', title: 'Residuals by Income Deciles', desc: 'Residual distribution across income deciles highlighting heteroskedasticity and under-prediction at the top decile.' },
  { id: 'fig08', file: 'fig08_ocean_proximity.png', title: 'Ocean Proximity Stratification', desc: 'Boxplots of house prices across categorical zones: Island, Near Bay, Near Ocean, <1H Ocean, and Inland.' },
  { id: 'fig09', file: 'fig09_moran_scatter.png', title: "Moran's I Spatial Autocorrelation", desc: "Moran scatter plot demonstrating strong spatial clustering (I = 0.79) among adjacent census districts." },
  { id: 'fig10', file: 'fig10_spatial_features_gain.png', title: 'Spatial Feature Engineering Gain', desc: 'Performance improvement obtained by engineering distance vectors to 5 major California urban employment hubs.' },
  { id: 'fig11', file: 'fig11_ratio_features.png', title: 'Engineered Ratio Distributions', desc: 'Distributions of rooms_per_household, bedrooms_ratio, and population_per_household normalized features.' },
  { id: 'fig12', file: 'fig12_model_comparison.png', title: 'Multi-Model Benchmark Comparison', desc: 'Cross-validated and test RMSE comparisons across Dummy, Ridge, k-NN, Random Forest, HistGB, and XGBoost.' },
  { id: 'fig13', file: 'fig13_pred_vs_actual.png', title: 'Predicted vs Actual Values', desc: 'Parity scatter plot showing tight clustering along the 45-degree line for XGBoost, with visible $500k ceiling.' },
  { id: 'fig14', file: 'fig14_learning_curve.png', title: 'XGBoost Learning Curves', desc: 'Train vs validation RMSE across training sample sizes, confirming no severe overfitting or high variance.' },
  { id: 'fig15', file: 'fig15_residual_diagnostics.png', title: 'Residual Diagnostic Plots', desc: 'Standardized residuals vs fitted values and Q-Q normality plot demonstrating homoskedasticity bounds.' },
  { id: 'fig16', file: 'fig16_segment_errors.png', title: 'Segmented Sub-group Error Analysis', desc: 'RMSE broken down by coastal zones, housing age tiers, and price brackets.' },
  { id: 'fig17', file: 'fig17_residual_map.png', title: 'Spatial Residual Error Map', desc: 'California geographic map colored by model residual errors, confirming elimination of regional spatial bias.' },
  { id: 'fig18', file: 'fig18_permutation_importance.png', title: 'Permutation Feature Importance', desc: 'Top predictors ranked by test RMSE degradation upon shuffling: median_income, distance_to_SF, and bedrooms_ratio.' },
  { id: 'fig19', file: 'fig19_shap_summary.png', title: 'SHAP Beeswarm Value Summary', desc: 'TreeSHAP global feature impact plot showing directionality and magnitude of each predictor on district valuations.' },
  { id: 'fig20', file: 'fig20_partial_dependence.png', title: 'Partial Dependence Plots (PDP)', desc: 'Marginal effects of median income, housing age, and geographic latitude on predicted house value.' }
];

let debounceTimer = null;

document.addEventListener('DOMContentLoaded', () => {
  initFormBindings();
  initPresetButtons();
  initGallery();
  initModal();
  checkHealth();

  // Run initial prediction
  calculatePrediction();
});

function initFormBindings() {
  const form = document.getElementById('predictionForm');
  const calculateBtn = document.getElementById('btnCalculate');
  const randomBtn = document.getElementById('btnRandomSample');
  const resetBtn = document.getElementById('btnResetForm');

  // Input & display sync
  bindRangeWithDisplay('inpMedianIncome', 'dispMedianIncome', val => `${parseFloat(val).toFixed(2)} (~$${(val * 10000).toLocaleString()}/yr)`);
  bindRangeWithDisplay('inpHousingAge', 'dispHousingAge', val => `${val} yrs`);
  bindRangeWithDisplay('inpTotalRooms', 'dispTotalRooms', val => parseInt(val).toLocaleString());
  bindRangeWithDisplay('inpTotalBedrooms', 'dispTotalBedrooms', val => parseInt(val).toLocaleString());
  bindRangeWithDisplay('inpPopulation', 'dispPopulation', val => parseInt(val).toLocaleString());
  bindRangeWithDisplay('inpHouseholds', 'dispHouseholds', val => parseInt(val).toLocaleString());

  const inpLong = document.getElementById('inpLongitude');
  const dispLong = document.getElementById('dispLongitude');
  inpLong.addEventListener('input', () => {
    dispLong.textContent = `${parseFloat(inpLong.value || 0).toFixed(2)}°`;
    triggerDebouncedPrediction();
  });

  const inpLat = document.getElementById('inpLatitude');
  const dispLat = document.getElementById('dispLatitude');
  inpLat.addEventListener('input', () => {
    dispLat.textContent = `${parseFloat(inpLat.value || 0).toFixed(2)}°`;
    triggerDebouncedPrediction();
  });

  // Radio pills
  document.querySelectorAll('input[name="ocean_proximity"]').forEach(radio => {
    radio.addEventListener('change', () => triggerDebouncedPrediction());
  });

  calculateBtn.addEventListener('click', () => calculatePrediction());

  randomBtn.addEventListener('click', loadRandomSample);
  resetBtn.addEventListener('click', () => {
    loadPreset('la_coastal');
    document.getElementById('groundTruthBox').classList.add('hidden');
    document.getElementById('badgeSampleState').textContent = 'Live Inference';
  });
}

function bindRangeWithDisplay(rangeId, dispId, formatFn) {
  const range = document.getElementById(rangeId);
  const disp = document.getElementById(dispId);

  range.addEventListener('input', () => {
    disp.textContent = formatFn(range.value);
    triggerDebouncedPrediction();
  });
}

function initPresetButtons() {
  document.querySelectorAll('.preset-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const presetKey = btn.dataset.preset;
      if (PRESETS[presetKey]) {
        loadPreset(presetKey);
        document.getElementById('groundTruthBox').classList.add('hidden');
        document.getElementById('badgeSampleState').textContent = 'Preset District';
        calculatePrediction();
      }
    });
  });
}

function loadPreset(presetKey) {
  const p = PRESETS[presetKey];
  if (!p) return;

  setInputValue('inpLongitude', p.longitude, `${p.longitude.toFixed(2)}°`, 'dispLongitude');
  setInputValue('inpLatitude', p.latitude, `${p.latitude.toFixed(2)}°`, 'dispLatitude');
  setInputValue('inpMedianIncome', p.median_income, `${p.median_income.toFixed(2)} (~$${(p.median_income * 10000).toLocaleString()}/yr)`, 'dispMedianIncome');
  setInputValue('inpHousingAge', p.housing_median_age, `${p.housing_median_age} yrs`, 'dispHousingAge');
  setInputValue('inpTotalRooms', p.total_rooms, p.total_rooms.toLocaleString(), 'dispTotalRooms');
  setInputValue('inpTotalBedrooms', p.total_bedrooms, p.total_bedrooms.toLocaleString(), 'dispTotalBedrooms');
  setInputValue('inpPopulation', p.population, p.population.toLocaleString(), 'dispPopulation');
  setInputValue('inpHouseholds', p.households, p.households.toLocaleString(), 'dispHouseholds');

  const radio = document.querySelector(`input[name="ocean_proximity"][value="${p.ocean_proximity}"]`);
  if (radio) radio.checked = true;
}

function setInputValue(id, val, dispVal, dispId) {
  const el = document.getElementById(id);
  el.value = val;
  if (dispId) {
    document.getElementById(dispId).textContent = dispVal;
  }
}

function triggerDebouncedPrediction() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    calculatePrediction();
  }, 250);
}

async function calculatePrediction() {
  const priceDisplay = document.getElementById('resultPrice');
  const ciRange = document.getElementById('resultCiRange');
  const featRooms = document.getElementById('featRoomsPerHh');
  const featBedrooms = document.getElementById('featBedroomsRatio');
  const featPop = document.getElementById('featPopPerHh');
  const featHub = document.getElementById('featHub');

  const payload = {
    longitude: parseFloat(document.getElementById('inpLongitude').value),
    latitude: parseFloat(document.getElementById('inpLatitude').value),
    housing_median_age: parseFloat(document.getElementById('inpHousingAge').value),
    total_rooms: parseFloat(document.getElementById('inpTotalRooms').value),
    total_bedrooms: parseFloat(document.getElementById('inpTotalBedrooms').value),
    population: parseFloat(document.getElementById('inpPopulation').value),
    households: parseFloat(document.getElementById('inpHouseholds').value),
    median_income: parseFloat(document.getElementById('inpMedianIncome').value),
    ocean_proximity: document.querySelector('input[name="ocean_proximity"]:checked').value
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error('Prediction request failed');
    const data = await res.json();

    priceDisplay.textContent = data.predicted_formatted;
    ciRange.textContent = `${data.confidence_band.lower_formatted} – ${data.confidence_band.upper_formatted}`;

    const fd = data.features_derived;
    featRooms.textContent = fd.rooms_per_household;
    featBedrooms.textContent = `${(fd.bedrooms_ratio * 100).toFixed(1)}%`;
    featPop.textContent = fd.population_per_household;
    featHub.textContent = `${fd.closest_urban_hub} (${fd.distance_to_hub_km} km)`;

  } catch (err) {
    console.error('Prediction API Error:', err);
    priceDisplay.textContent = 'Valuation Error';
  }
}

async function loadRandomSample() {
  try {
    const res = await fetch('/api/sample');
    if (!res.ok) throw new Error('Failed to load sample');
    const samples = await res.json();
    if (!samples || samples.length === 0) return;

    const row = samples[Math.floor(Math.random() * samples.length)];

    setInputValue('inpLongitude', row.longitude, `${row.longitude.toFixed(2)}°`, 'dispLongitude');
    setInputValue('inpLatitude', row.latitude, `${row.latitude.toFixed(2)}°`, 'dispLatitude');
    setInputValue('inpMedianIncome', row.median_income, `${row.median_income.toFixed(2)} (~$${(row.median_income * 10000).toLocaleString()}/yr)`, 'dispMedianIncome');
    setInputValue('inpHousingAge', row.housing_median_age, `${row.housing_median_age} yrs`, 'dispHousingAge');
    setInputValue('inpTotalRooms', row.total_rooms, row.total_rooms.toLocaleString(), 'dispTotalRooms');
    setInputValue('inpTotalBedrooms', row.total_bedrooms || Math.round(row.total_rooms * 0.2), (row.total_bedrooms || 0).toLocaleString(), 'dispTotalBedrooms');
    setInputValue('inpPopulation', row.population, row.population.toLocaleString(), 'dispPopulation');
    setInputValue('inpHouseholds', row.households, row.households.toLocaleString(), 'dispHouseholds');

    const radio = document.querySelector(`input[name="ocean_proximity"][value="${row.ocean_proximity}"]`);
    if (radio) radio.checked = true;

    document.querySelectorAll('.preset-pill').forEach(b => b.classList.remove('active'));

    // Trigger prediction
    await calculatePrediction();

    // Show actual comparison box
    const actual = row.median_house_value;
    if (actual) {
      const gtBox = document.getElementById('groundTruthBox');
      gtBox.classList.remove('hidden');
      document.getElementById('gtActual').textContent = `$${actual.toLocaleString()}`;

      const predText = document.getElementById('resultPrice').textContent.replace(/[$,]/g, '');
      const pred = parseFloat(predText);
      const diff = pred - actual;
      const pct = ((diff / actual) * 100).toFixed(1);

      const gtDiffEl = document.getElementById('gtDiff');
      gtDiffEl.textContent = `${diff >= 0 ? '+' : ''}$${Math.round(diff).toLocaleString()} (${pct}%)`;
      gtDiffEl.className = Math.abs(pct) <= 20 ? 'gt-val highlight-green' : 'gt-val';

      document.getElementById('badgeSampleState').textContent = 'Test Dataset Sample';
    }

  } catch (err) {
    console.error('Error loading sample:', err);
  }
}

function initGallery() {
  const container = document.getElementById('galleryGrid');
  container.innerHTML = '';

  FIGURES.forEach(fig => {
    const card = document.createElement('div');
    card.className = 'gallery-card';
    card.innerHTML = `
      <div class="gallery-card-img-wrap">
        <img src="/figures/${fig.file}" alt="${fig.title}" loading="lazy">
      </div>
      <div class="gallery-card-body">
        <span class="fig-id">${fig.id.toUpperCase()}</span>
        <h5 class="fig-name">${fig.title}</h5>
        <p class="fig-desc">${fig.desc}</p>
      </div>
    `;
    card.addEventListener('click', () => openModal(fig));
    container.appendChild(card);
  });
}

function initModal() {
  const modal = document.getElementById('figureModal');
  const closeBtn = document.getElementById('modalClose');

  closeBtn.addEventListener('click', closeModal);
  modal.addEventListener('click', e => {
    if (e.target === modal) closeModal();
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
      closeModal();
    }
  });
}

function openModal(fig) {
  const modal = document.getElementById('figureModal');
  document.getElementById('modalTitle').textContent = `${fig.id.toUpperCase()}: ${fig.title}`;
  document.getElementById('modalImg').src = `/figures/${fig.file}`;
  document.getElementById('modalDesc').textContent = fig.desc;
  modal.classList.remove('hidden');
}

function closeModal() {
  document.getElementById('figureModal').classList.add('hidden');
}

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.status === 'healthy') {
      document.getElementById('modelStatus').textContent = 'XGBoost Active (Port 5050)';
    }
  } catch (e) {
    document.getElementById('modelStatus').textContent = 'API Reconnecting...';
  }
}
