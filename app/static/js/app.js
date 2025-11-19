// Global variables
let currentUploadId = null;
let currentTrainingId = null;

console.log('=== AutoML App.js loaded successfully ===');

// DOM elements - will be initialized when DOM is ready
let uploadArea, fileInput, uploadSpinner, dataOverviewSection, trainingConfigSection, trainingProgressSection, resultsSection, errorAlert;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('=== DOM Content Loaded ===');
    
    // Initialize DOM elements
    uploadArea = document.getElementById('uploadArea');
    fileInput = document.getElementById('fileInput');
    uploadSpinner = document.getElementById('uploadSpinner');
    dataOverviewSection = document.getElementById('data-overview-section');
    trainingConfigSection = document.getElementById('training-config-section');
    trainingProgressSection = document.getElementById('training-progress-section');
    resultsSection = document.getElementById('results-section');
    errorAlert = document.getElementById('errorAlert');
    
    console.log('DOM elements initialized:', {
        uploadArea: !!uploadArea,
        fileInput: !!fileInput,
        dataOverviewSection: !!dataOverviewSection,
        qualityBtn: !!document.getElementById('viewQualityReportBtn')
    });
    
    setupEventListeners();
    setupFormValidation();
});

function setupEventListeners() {
    // File upload drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleFileDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    // EDA generation - add error handling
    const edaBtn = document.getElementById('generateEDABtn');
    if (edaBtn) {
        edaBtn.addEventListener('click', generateEDA);
    }
    
    // Data Quality Report - add error handling and debugging
    const qualityBtn = document.getElementById('viewQualityReportBtn');
    if (qualityBtn) {
        console.log('Quality button found, adding event listener');
        qualityBtn.addEventListener('click', function(event) {
            console.log('Quality button clicked!');
            event.preventDefault();
            viewQualityReport();
        });
    } else {
        console.log('Quality button NOT found during setup');
    }

    // Training form
    const trainingForm = document.getElementById('trainingForm');
    if (trainingForm) {
        trainingForm.addEventListener('submit', startTraining);
    }

    // Test size slider
    const testSizeSlider = document.getElementById('testSize');
    testSizeSlider.addEventListener('input', function() {
        document.getElementById('testSizeValue').textContent = Math.round(this.value * 100) + '%';
    });
}

function setupFormValidation() {
    // Form validation logic can be added here
}

// File upload handling
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleFileDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        uploadFile(e.target.files[0]);
    }
}

async function uploadFile(file) {
    // Validate file
    const allowedExtensions = ['.csv', '.xlsx', '.xls'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedExtensions.includes(fileExtension)) {
        showError('Please select a CSV or Excel file');
        return;
    }

    // Show loading
    uploadSpinner.style.display = 'block';
    hideError();

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/upload/', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            currentUploadId = result.upload_id;
            showDataOverview(result.dataset_info);
            updateStepIndicator(2);
        } else {
            showError(result.detail || 'Upload failed');
        }
    } catch (error) {
        showError('Upload failed: ' + error.message);
    } finally {
        uploadSpinner.style.display = 'none';
    }
}

function showDataOverview(datasetInfo) {
    const datasetInfoDiv = document.getElementById('datasetInfo');
    
    // Create dataset overview
    const overview = `
        <div class="row">
            <div class="col-md-6">
                <h5>Dataset Information</h5>
                <table class="table table-sm">
                    <tr><td><strong>Filename:</strong></td><td>${datasetInfo.filename}</td></tr>
                    <tr><td><strong>Rows:</strong></td><td>${datasetInfo.shape[0].toLocaleString()}</td></tr>
                    <tr><td><strong>Columns:</strong></td><td>${datasetInfo.shape[1]}</td></tr>
                </table>
            </div>
            <div class="col-md-6">
                <h5>Missing Data</h5>
                <div class="small">
                    ${Object.entries(datasetInfo.missing_values)
                        .filter(([col, count]) => count > 0)
                        .map(([col, count]) => `<div>${col}: ${count} missing</div>`)
                        .join('') || '<div class="text-success">No missing values found</div>'}
                </div>
            </div>
        </div>
        <div class="mt-3">
            <h5>Column Types</h5>
            <div class="row">
                ${datasetInfo.columns.map(col => `
                    <div class="col-md-4 mb-2">
                        <span class="badge bg-${getColumnTypeBadgeColor(datasetInfo.column_types[col])} me-1">
                            ${datasetInfo.column_types[col]}
                        </span>
                        ${col}
                    </div>
                `).join('')}
            </div>
        </div>
        <div class="mt-3">
            <h5>Target Column Suggestions</h5>
            <div>
                ${datasetInfo.target_suggestions.map(col => `
                    <span class="badge bg-info me-1">${col}</span>
                `).join('') || '<span class="text-muted">No suggestions available</span>'}
            </div>
        </div>
    `;
    
    datasetInfoDiv.innerHTML = overview;
    
    // Populate target column dropdown
    const targetSelect = document.getElementById('targetColumn');
    targetSelect.innerHTML = '<option value="">Select target column...</option>';
    datasetInfo.columns.forEach(col => {
        const option = new Option(col, col);
        if (datasetInfo.target_suggestions.includes(col)) {
            option.classList.add('fw-bold');
        }
        targetSelect.appendChild(option);
    });

    // Set suggested problem type
    if (datasetInfo.problem_type) {
        document.getElementById('problemType').value = datasetInfo.problem_type;
    }
    
    dataOverviewSection.style.display = 'block';
    
    // Re-attach event listeners after the section becomes visible
    console.log('Data overview section shown, re-checking quality button');
    const qualityBtn = document.getElementById('viewQualityReportBtn');
    if (qualityBtn) {
        console.log('Quality button is now visible and accessible');
        // Remove any existing listeners and add new one
        qualityBtn.onclick = function(event) {
            console.log('Quality button clicked via onclick!');
            event.preventDefault();
            viewQualityReport();
        };
    } else {
        console.log('Quality button still not found after showing data overview');
    }
}

function getColumnTypeBadgeColor(type) {
    switch (type) {
        case 'numerical': return 'primary';
        case 'categorical': return 'success';
        case 'datetime': return 'warning';
        case 'text': return 'info';
        default: return 'secondary';
    }
}

async function generateEDA() {
    if (!currentUploadId) return;

    const btn = document.getElementById('generateEDABtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<div class="spinner-border spinner-border-sm me-1"></div>Generating...';
    btn.disabled = true;

    try {
        const response = await fetch(`/api/train/eda/${currentUploadId}`);
        const result = await response.json();

        if (response.ok) {
            showEDAResults(result);
            updateStepIndicator(3);
            trainingConfigSection.style.display = 'block';
        } else {
            showError(result.detail || 'EDA generation failed');
        }
    } catch (error) {
        showError('EDA generation failed: ' + error.message);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

// Data Quality Report function
async function viewQualityReport() {
    console.log('viewQualityReport function called');
    console.log('currentUploadId:', currentUploadId);
    
    if (!currentUploadId) {
        console.log('No currentUploadId, showing error');
        showError('No dataset uploaded');
        return;
    }

    const btn = document.getElementById('viewQualityReportBtn');
    console.log('Quality button element:', btn);
    
    if (!btn) {
        console.log('Button not found!');
        showError('Button not found');
        return;
    }

    const originalText = btn.innerHTML;
    btn.innerHTML = '<div class="spinner-border spinner-border-sm me-1"></div>Generating Report...';
    btn.disabled = true;

    try {
        // Open quality report in new window/tab
        const reportUrl = `/api/quality/report/${currentUploadId}`;
        console.log('Attempting to open URL:', reportUrl);
        const reportWindow = window.open(reportUrl, '_blank', 'width=1200,height=800,scrollbars=yes');
        
        if (!reportWindow || reportWindow.closed || typeof reportWindow.closed == 'undefined') {
            console.log('Popup blocked or failed to open');
            // Show message instead of trying modal
            showSuccessMessage('Please allow popups for this site, then try again. Your browser blocked the quality report window.');
        } else {
            console.log('Report window opened successfully');
            // Show success message
            showSuccessMessage('Data Quality Report opened in new tab');
        }
    } catch (error) {
        console.error('Error opening quality report:', error);
        showError('Failed to open quality report: ' + error.message);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

// Alternative modal approach for quality report
async function showQualityReportModal() {
    try {
        const response = await fetch(`/api/quality/report/${currentUploadId}/json`);
        const qualityData = await response.json();

        if (response.ok) {
            displayQualityReportInModal(qualityData);
        } else {
            showError('Failed to load quality report data');
        }
    } catch (error) {
        showError('Failed to load quality report: ' + error.message);
    }
}

// Display quality report in a modal
function displayQualityReportInModal(qualityData) {
    const modalHtml = `
        <div class="modal fade" id="qualityReportModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header bg-primary text-white">
                        <h5 class="modal-title">
                            <i class="fas fa-clipboard-check me-2"></i>Data Quality Report
                        </h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row mb-4">
                            <div class="col-md-3">
                                <div class="text-center p-3 bg-light rounded">
                                    <h2 class="display-4 text-primary">${qualityData.data_quality_score.overall_score}</h2>
                                    <p class="mb-0">Quality Score</p>
                                    <small class="text-muted">${qualityData.data_quality_score.grade}</small>
                                </div>
                            </div>
                            <div class="col-md-9">
                                <div class="row">
                                    <div class="col-6 col-md-3">
                                        <div class="text-center p-2">
                                            <h4>${qualityData.dataset_overview.total_rows.toLocaleString()}</h4>
                                            <small class="text-muted">Total Rows</small>
                                        </div>
                                    </div>
                                    <div class="col-6 col-md-3">
                                        <div class="text-center p-2">
                                            <h4>${qualityData.dataset_overview.total_columns}</h4>
                                            <small class="text-muted">Columns</small>
                                        </div>
                                    </div>
                                    <div class="col-6 col-md-3">
                                        <div class="text-center p-2">
                                            <h4>${qualityData.missing_data_analysis.missing_data_percentage}%</h4>
                                            <small class="text-muted">Missing Data</small>
                                        </div>
                                    </div>
                                    <div class="col-6 col-md-3">
                                        <div class="text-center p-2">
                                            <h4>${qualityData.dataset_overview.duplicate_percentage}%</h4>
                                            <small class="text-muted">Duplicates</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        ${generateMissingDataSection(qualityData.missing_data_analysis)}
                        ${generateColumnAnalysisSection(qualityData.column_analysis)}
                        ${generateRecommendationsSection(qualityData.recommendations)}
                    </div>
                    <div class="modal-footer">
                        <a href="/api/quality/report/${currentUploadId}" target="_blank" class="btn btn-primary">
                            <i class="fas fa-external-link-alt me-1"></i>Open Full Report
                        </a>
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if present
    const existingModal = document.getElementById('qualityReportModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to DOM and show it
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = new bootstrap.Modal(document.getElementById('qualityReportModal'));
    modal.show();
}

// Helper functions for modal content
function generateMissingDataSection(missingData) {
    if (!missingData.columns_missing_analysis || missingData.columns_missing_analysis.length === 0) {
        return '<div class="alert alert-success"><i class="fas fa-check-circle me-2"></i>No missing data detected!</div>';
    }

    const tableRows = missingData.columns_missing_analysis.map(col => `
        <tr>
            <td><strong>${col.column}</strong></td>
            <td>${col.missing_count.toLocaleString()}</td>
            <td>${col.missing_percentage}%</td>
            <td><span class="badge bg-${getSeverityColor(col.severity)}">${col.severity}</span></td>
        </tr>
    `).join('');

    return `
        <div class="mb-4">
            <h6><i class="fas fa-exclamation-triangle me-2"></i>Missing Data Analysis</h6>
            <table class="table table-sm">
                <thead><tr><th>Column</th><th>Missing Count</th><th>Missing %</th><th>Severity</th></tr></thead>
                <tbody>${tableRows}</tbody>
            </table>
        </div>
    `;
}

function generateColumnAnalysisSection(columnAnalysis) {
    const tableRows = columnAnalysis.slice(0, 10).map(col => `
        <tr>
            <td><strong>${col.column_name}</strong></td>
            <td><span class="badge bg-${getTypeColor(col.detected_type)}">${col.detected_type}</span></td>
            <td>${col.unique_values.toLocaleString()}</td>
            <td><span class="text-${getCardinalityColor(col.cardinality)}">${col.cardinality}</span></td>
            <td>${col.missing_percentage}%</td>
        </tr>
    `).join('');

    return `
        <div class="mb-4">
            <h6><i class="fas fa-table me-2"></i>Column Analysis (Top 10)</h6>
            <table class="table table-sm">
                <thead><tr><th>Column</th><th>Type</th><th>Unique</th><th>Cardinality</th><th>Missing %</th></tr></thead>
                <tbody>${tableRows}</tbody>
            </table>
        </div>
    `;
}

function generateRecommendationsSection(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        return '<div class="alert alert-info"><i class="fas fa-info-circle me-2"></i>No specific recommendations at this time.</div>';
    }

    const recommendationItems = recommendations.slice(0, 5).map(rec => `
        <li class="list-group-item"><i class="fas fa-lightbulb me-2 text-warning"></i>${rec}</li>
    `).join('');

    return `
        <div class="mb-4">
            <h6><i class="fas fa-lightbulb me-2"></i>Data Quality Recommendations</h6>
            <ul class="list-group list-group-flush">${recommendationItems}</ul>
        </div>
    `;
}

// Helper functions for styling
function getSeverityColor(severity) {
    const colors = { 'Critical': 'danger', 'High': 'warning', 'Medium': 'info', 'Low': 'success' };
    return colors[severity] || 'secondary';
}

function getTypeColor(type) {
    const colors = { 'numerical': 'success', 'categorical': 'info', 'text': 'warning' };
    return colors[type] || 'secondary';
}

function getCardinalityColor(cardinality) {
    const colors = { 'Low': 'success', 'Medium': 'warning', 'High': 'danger' };
    return colors[cardinality] || 'muted';
}

// Success message function
function showSuccessMessage(message) {
    const alertHtml = `
        <div class="alert alert-success alert-dismissible fade show mt-3" role="alert">
            <i class="fas fa-check-circle me-2"></i>${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    const container = document.querySelector('.container');
    const existingAlert = container.querySelector('.alert-success');
    if (existingAlert) existingAlert.remove();
    
    container.insertAdjacentHTML('afterbegin', alertHtml);
    
    // Auto-dismiss after 3 seconds
    setTimeout(() => {
        const alert = container.querySelector('.alert-success');
        if (alert) alert.remove();
    }, 3000);
}

function showEDAResults(edaData) {
    const edaResults = document.getElementById('edaResults');
    
    let html = '<h5>Exploratory Data Analysis</h5>';
    
    // Summary statistics
    if (edaData.summary_stats) {
        html += `
            <div class="row">
                <div class="col-md-6">
                    <h6>Numerical Columns</h6>
                    ${edaData.summary_stats.numerical_columns.map(col => `
                        <div class="metric-badge">
                            <strong>${col.name}</strong><br>
                            Mean: ${col.mean?.toFixed(2) || 'N/A'}<br>
                            Std: ${col.std?.toFixed(2) || 'N/A'}
                        </div>
                    `).join('')}
                </div>
                <div class="col-md-6">
                    <h6>Categorical Columns</h6>
                    ${edaData.summary_stats.categorical_columns.map(col => `
                        <div class="metric-badge">
                            <strong>${col.name}</strong><br>
                            Unique: ${col.unique_values}<br>
                            Most frequent: ${col.most_frequent || 'N/A'}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    // Visualizations
    if (edaData.visualizations && edaData.visualizations.length > 0) {
        html += '<h6 class="mt-4">Data Visualizations</h6>';
        edaData.visualizations.forEach((viz, index) => {
            html += `<div class="plot-container" id="plot-${index}" style="width:100%; height:400px;"></div>`;
        });
    }
    
    edaResults.innerHTML = html;
    
    // Render plots with a small delay to ensure DOM is ready
    if (edaData.visualizations) {
        setTimeout(() => {
            edaData.visualizations.forEach((viz, index) => {
                try {
                    const plotData = JSON.parse(viz.data);
                    const config = {
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                    };
                    
                    // Ensure layout has proper sizing
                    const layout = {
                        ...plotData.layout,
                        autosize: true,
                        margin: { t: 50, r: 50, b: 50, l: 50 }
                    };
                    
                    Plotly.newPlot(`plot-${index}`, plotData.data, layout, config);
                } catch (error) {
                    console.error('Error rendering plot:', error);
                    // Show error message in the plot container
                    document.getElementById(`plot-${index}`).innerHTML = 
                        '<div class="alert alert-warning">Failed to render visualization</div>';
                }
            });
        }, 100);
    }
}

async function startTraining(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const trainingRequest = {
        upload_id: currentUploadId,
        target_column: document.getElementById('targetColumn').value,
        problem_type: document.getElementById('problemType').value,
        test_size: parseFloat(document.getElementById('testSize').value),
        cv_folds: parseInt(document.getElementById('cvFolds').value)
    };

    // Show training progress
    updateStepIndicator(4);
    trainingProgressSection.style.display = 'block';
    hideError();

    try {
        const response = await fetch('/api/train/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(trainingRequest)
        });

        const result = await response.json();

        if (response.ok) {
            currentTrainingId = result.training_id;
            showTrainingResults(result);
            updateStepIndicator(5);
        } else {
            showError(result.detail || 'Training failed');
        }
    } catch (error) {
        showError('Training failed: ' + error.message);
    } finally {
        trainingProgressSection.style.display = 'none';
    }
}

function showTrainingResults(trainingData) {
    const resultsDiv = document.getElementById('trainingResults');
    
    // Create results overview
    let html = `
        <div class="alert alert-success">
            <h5><i class="fas fa-check-circle me-2"></i>Training Completed Successfully!</h5>
            <p>Best Model: <strong>${trainingData.best_model}</strong></p>
            <p>Problem Type: <strong>${trainingData.problem_type}</strong></p>
            <p>Models Trained: <strong>${trainingData.model_performances.length}</strong></p>
        </div>
        
        <h5>Model Performance Comparison</h5>
        <div class="row">
    `;
    
    // Model performance cards
    trainingData.model_performances.forEach(model => {
        const primaryMetric = trainingData.problem_type === 'classification' ? 'accuracy' : 'r2_score';
        const primaryValue = model.metrics[primaryMetric] || model.metrics[Object.keys(model.metrics)[0]];
        
        html += `
            <div class="col-md-6 col-lg-4">
                <div class="model-card ${model.is_best ? 'best-model' : ''}">
                    <h6>${model.model_name} ${model.is_best ? '<i class="fas fa-crown text-warning"></i>' : ''}</h6>
                    <div class="metric-badge">
                        <strong>${primaryMetric}:</strong> ${(primaryValue * 100).toFixed(2)}%
                    </div>
                    <div class="small text-muted">
                        Training Time: ${model.training_time.toFixed(2)}s
                    </div>
                    <div class="mt-2">
                        ${Object.entries(model.metrics).slice(0, 3).map(([key, value]) => 
                            `<span class="metric-badge">${key}: ${(value * 100).toFixed(1)}%</span>`
                        ).join('')}
                    </div>
                </div>
            </div>
        `;
    });
    
    html += `
        </div>
        
        <div class="mt-4">
            <h5>Actions</h5>
            <div class="btn-group" role="group">
                <button class="btn btn-primary" onclick="loadFeatureImportance()">
                    <i class="fas fa-chart-bar me-1"></i>Feature Importance
                </button>
                <button class="btn btn-success" onclick="downloadModel()">
                    <i class="fas fa-download me-1"></i>Download Model
                </button>
                <button class="btn btn-info" onclick="downloadReport()">
                    <i class="fas fa-file-alt me-1"></i>Download Report
                </button>
                <button class="btn btn-warning" onclick="viewQualityReport()">
                    <i class="fas fa-clipboard-check me-1"></i>Data Quality Report
                </button>
            </div>
        </div>
        
        <div id="featureImportanceSection" class="mt-4" style="display: none;">
            <!-- Feature importance will be loaded here -->
        </div>
    `;
    
    resultsDiv.innerHTML = html;
    resultsSection.style.display = 'block';
}

async function loadFeatureImportance() {
    if (!currentTrainingId) return;

    const section = document.getElementById('featureImportanceSection');
    section.innerHTML = '<div class="text-center"><div class="spinner-border"></div></div>';
    section.style.display = 'block';

    try {
        const response = await fetch(`/api/evaluate/feature-importance/${currentTrainingId}`);
        const result = await response.json();

        if (response.ok) {
            let html = '<h6>Feature Importance</h6>';
            if (result.chart) {
                html += '<div id="featureImportancePlot" style="width:100%; height:500px;"></div>';
                section.innerHTML = html;
                
                // Render the plot with proper sizing
                setTimeout(() => {
                    try {
                        const plotData = JSON.parse(result.chart);
                        const config = {
                            responsive: true,
                            displayModeBar: true,
                            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                        };
                        
                        const layout = {
                            ...plotData.layout,
                            autosize: true,
                            margin: { t: 50, r: 50, b: 50, l: 100 }
                        };
                        
                        Plotly.newPlot('featureImportancePlot', plotData.data, layout, config);
                    } catch (error) {
                        console.error('Error rendering feature importance plot:', error);
                        document.getElementById('featureImportancePlot').innerHTML = 
                            '<div class="alert alert-warning">Failed to render feature importance plot</div>';
                    }
                }, 100);
            } else {
                html += '<div class="alert alert-warning">Feature importance not available for this model type.</div>';
                section.innerHTML = html;
            }
        } else {
            section.innerHTML = '<div class="alert alert-danger">Error loading feature importance</div>';
        }
    } catch (error) {
        section.innerHTML = '<div class="alert alert-danger">Error loading feature importance</div>';
    }
}

function downloadModel() {
    if (!currentTrainingId) return;
    window.open(`/api/download/model/${currentTrainingId}`, '_blank');
}

function downloadReport() {
    if (!currentTrainingId) return;
    window.open(`/api/download/report/${currentTrainingId}`, '_blank');
}

function updateStepIndicator(activeStep) {
    // Remove all active/completed classes
    for (let i = 1; i <= 5; i++) {
        const step = document.getElementById(`step${i}`);
        step.classList.remove('active', 'completed');
        
        if (i < activeStep) {
            step.classList.add('completed');
        } else if (i === activeStep) {
            step.classList.add('active');
        }
    }
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    errorAlert.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

function hideError() {
    errorAlert.style.display = 'none';
}

// Utility functions
function formatNumber(num) {
    if (typeof num !== 'number') return 'N/A';
    return num.toFixed(2);
}

function formatPercentage(num) {
    if (typeof num !== 'number') return 'N/A';
    return (num * 100).toFixed(1) + '%';
}