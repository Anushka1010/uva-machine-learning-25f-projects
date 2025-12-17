# Grafana Dashboards for Credit Model Monitoring

This directory contains Grafana dashboard configurations for comprehensive monitoring of the Fair Credit Score Prediction system.

## Dashboards Overview

### 1. Model Performance Dashboard
**File:** `dashboards/model_performance.json`

Monitors model prediction performance and latency:
- **Model Accuracy**: Current model accuracy gauge
- **Cross-Validation Score**: Mean CV score with thresholds
- **CV Score Standard Deviation**: Consistency metric
- **Prediction Rate**: Requests per minute by outcome and group
- **Prediction Latency**: p50, p95, p99 latency percentiles
- **Prediction Errors**: Error rate by type
- **Predictions by Group**: Distribution visualization
- **Predictions by Outcome**: Approved vs denied breakdown

**Alerts:**
- High latency (p95 > 0.5s)
- High error rate (> 1% of requests)

### 2. Fairness Metrics Dashboard
**File:** `dashboards/fairness_metrics.json`

Tracks regulatory compliance and fairness:
- **Disparate Impact Ratio**: Four-Fifths Rule compliance (target ≥ 0.80)
- **Statistical Parity Difference**: Group approval rate difference (target < 0.05)
- **Equalized Odds Difference**: TPR/FPR parity (target < 0.10)
- **Fairness Metrics Over Time**: Trend analysis
- **Approval Rate by Group**: Compare privileged vs unprivileged
- **Prediction Volume by Group**: Traffic distribution
- **Fairness Compliance Table**: At-a-glance status

**Key Targets:**
- Disparate Impact ≥ 0.80
- Statistical Parity < 0.05
- Equalized Odds < 0.10

### 3. Data Quality Dashboard
**File:** `dashboards/data_quality.json`

Monitors input data quality and validation:
- **Overall Data Quality Score**: 0-100 score based on validation
- **Missing Values**: Count of missing fields
- **Validation Issues Rate**: Issues per second
- **Unseen Categories Rate**: New categorical values encountered
- **Quality Score Over Time**: Historical trends
- **Validation Issues by Type**: Breakdown by issue category
- **Unseen Categories by Column**: Which features have drift
- **Missing Values Trend**: Track missing data patterns
- **Data Quality Issues Summary**: Tabular summary

**Alerts:**
- High validation issue rate (> 0.5/sec)

### 4. Calibration & Performance Dashboard
**File:** `dashboards/calibration.json`

Tracks model calibration and performance:
- **Model Calibration Status**: Whether calibration is enabled
- **Brier Score**: Probability calibration quality (lower is better)
- **CV Score Summary**: Training validation metrics
- **Brier Score Over Time**: Calibration drift detection
- **Feature Drift Detection**: Distribution shift monitoring
- **Model Performance Summary Table**: All metrics at a glance
- **Calibration Recommendations**: Actionable guidance

**Brier Score Interpretation:**
- < 0.10: Excellent
- 0.10-0.15: Good
- 0.15-0.20: Acceptable
- \> 0.20: Poor - recalibrate

**Alerts:**
- Poor calibration (Brier score > 0.20)

## Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Running FastAPI application with `/metrics` endpoint
- Prometheus configured to scrape the API

### Option 1: Docker Compose (Recommended)

1. **Start the monitoring stack:**

```bash
cd grafana
docker-compose up -d
```

This starts:
- Prometheus on port 9090
- Grafana on port 3000

2. **Access Grafana:**

Navigate to http://localhost:3000

- Default login: `admin` / `admin`
- Change password on first login

3. **Dashboards are automatically loaded** from the `dashboards/` directory

### Option 2: Manual Setup

1. **Install Grafana:**

```bash
# macOS
brew install grafana

# Ubuntu/Debian
sudo apt-get install grafana

# Or use Docker
docker run -d \
  -p 3000:3000 \
  -v $(pwd)/dashboards:/etc/grafana/dashboards \
  -v $(pwd)/provisioning:/etc/grafana/provisioning \
  --name=grafana \
  grafana/grafana
```

2. **Configure Prometheus datasource:**

- Navigate to Configuration → Data Sources
- Add Prometheus
- URL: `http://prometheus:9090` (Docker) or `http://localhost:9090`
- Save & Test

3. **Import dashboards:**

For each dashboard JSON file:
- Navigate to Dashboards → Import
- Upload the JSON file or paste contents
- Select Prometheus datasource
- Import

### Option 3: Provisioning (Production)

Dashboards and datasources can be auto-provisioned:

1. **Copy provisioning files:**

```bash
cp grafana/provisioning/datasources/prometheus.yaml /etc/grafana/provisioning/datasources/
cp grafana/provisioning/dashboards/dashboards.yaml /etc/grafana/provisioning/dashboards/
```

2. **Configure dashboard path in `dashboards.yaml`:**

Update the `path` to point to your `dashboards/` directory.

3. **Restart Grafana:**

```bash
sudo systemctl restart grafana-server
```

## Prometheus Configuration

Ensure Prometheus is configured to scrape your API metrics:

**prometheus.yml:**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'credit-api'
    static_configs:
      - targets: ['localhost:8000']  # Your API host:port
    metrics_path: '/metrics'
```

## Alert Configuration

### Prometheus Alertmanager Rules

Create `prometheus/alert_rules.yml`:

```yaml
groups:
  - name: credit_model_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighPredictionErrorRate
        expr: rate(prediction_errors_total[5m]) > 0.01
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction error rate"
          description: "Error rate is {{ $value }} errors/sec"

      # Poor calibration
      - alert: PoorModelCalibration
        expr: model_calibration_brier_score > 0.20
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model calibration has degraded"
          description: "Brier score is {{ $value }}, should be < 0.20"

      # Fairness violation
      - alert: DisparateImpactViolation
        expr: fairness_disparate_impact < 0.80
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Disparate impact below threshold"
          description: "Ratio is {{ $value }}, should be >= 0.80"

      # High latency
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency"
          description: "p95 latency is {{ $value }}s"

      # Data quality issues
      - alert: DataQualityDegradation
        expr: data_quality_score < 60
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Data quality score is low"
          description: "Quality score is {{ $value }}/100"
```

### Grafana Alert Channels

Configure notification channels in Grafana:

1. **Navigate to:** Alerting → Notification channels
2. **Add channel** (examples):
   - **Slack**: Webhook URL
   - **Email**: SMTP settings
   - **PagerDuty**: Integration key
   - **Webhook**: Custom endpoint

3. **Test** the channel
4. **Link to dashboards**: Alerts are pre-configured in dashboards

## Customization

### Adjusting Thresholds

Edit the JSON files to change thresholds:

```json
"thresholds": {
  "mode": "absolute",
  "steps": [
    {"value": 0, "color": "red"},
    {"value": 0.8, "color": "green"}  // Change this
  ]
}
```

### Adding Panels

1. Edit the dashboard in Grafana UI
2. Add/modify panels
3. **Export JSON**:
   - Dashboard settings → JSON Model
   - Copy and save to `dashboards/` directory

### Custom Queries

All queries use PromQL. Examples:

```promql
# Approval rate for unprivileged group
sum(rate(predictions_total{outcome="approved",group="unprivileged"}[5m]))
/
sum(rate(predictions_total{group="unprivileged"}[5m]))

# Error rate percentage
rate(prediction_errors_total[5m]) / rate(predictions_total[5m]) * 100

# Latency p99
histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[5m]))
```

## Troubleshooting

### No data in dashboards

**Check:**
1. Is the API running? `curl http://localhost:8000/health`
2. Are metrics exposed? `curl http://localhost:8000/metrics`
3. Is Prometheus scraping? Check targets at http://localhost:9090/targets
4. Is the datasource configured correctly in Grafana?

### Dashboards not loading

**Solutions:**
1. Check Grafana logs: `docker logs grafana` or `journalctl -u grafana-server`
2. Verify provisioning path in `dashboards.yaml`
3. Ensure JSON files are valid (use a JSON validator)
4. Restart Grafana after configuration changes

### Alerts not firing

**Check:**
1. Alert rules configured in Prometheus
2. Notification channels configured in Grafana
3. Alert conditions are being met
4. Check alerting tab in Grafana dashboard

## Maintenance

### Regular Tasks

1. **Weekly:**
   - Review all dashboards for anomalies
   - Check alert history

2. **Monthly:**
   - Update alert thresholds based on trends
   - Archive old dashboard versions

3. **Quarterly:**
   - Review and optimize queries
   - Add new metrics as needed

### Dashboard Versioning

Track dashboard changes:

```bash
# Commit dashboard changes
git add grafana/dashboards/*.json
git commit -m "Update fairness dashboard thresholds"
```

## Best Practices

1. **Start with default dashboards**, customize gradually
2. **Set up alerts** for critical metrics
3. **Document threshold changes** in git commits
4. **Test alerts** regularly (create test scenarios)
5. **Use folders** to organize dashboards in production
6. **Enable dashboard versioning** in Grafana
7. **Set up regular backups** of Grafana database

## Resources

- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Query Examples](https://prometheus.io/docs/prometheus/latest/querying/examples/)
- [PromQL Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Alerting Guide](https://grafana.com/docs/grafana/latest/alerting/)

## Support

For issues or questions:
1. Check the main project README
2. Review Grafana and Prometheus logs
3. Consult the official documentation
4. Open an issue in the project repository
