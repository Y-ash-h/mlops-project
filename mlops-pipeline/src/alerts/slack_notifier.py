"""Slack notification module for MLOps pipeline alerts."""
import json
import os
from typing import Optional

import requests


SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK")


def send_slack(text: str, channel: Optional[str] = None, blocks: Optional[list] = None) -> bool:
    """Send a message to Slack via webhook.
    
    Args:
        text: Message text
        channel: Optional channel override
        blocks: Optional Slack blocks for rich formatting
        
    Returns:
        True if message sent successfully, False if webhook not configured or error occurred
    """
    webhook = SLACK_WEBHOOK
    if not webhook:
        print("SLACK_WEBHOOK env var not set, skipping Slack notification")
        return False
    
    payload = {"text": text}
    if channel:
        payload["channel"] = channel
    if blocks:
        payload["blocks"] = blocks
    
    try:
        resp = requests.post(webhook, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"Failed to send Slack notification: {e}")
        return False


def alert_validation_fail(run_id: str, model_name: str, metric_name: str, metric_value: float) -> bool:
    """Alert on validation failure.
    
    Args:
        run_id: MLflow run ID
        model_name: Name of model that failed validation
        metric_name: Name of metric that failed threshold
        metric_value: Value of failed metric
        
    Returns:
        True if alert sent successfully
    """
    text = (
        f":warning: Validation failed for model *{model_name}* "
        f"(run: `{run_id}`). {metric_name}={metric_value:.4f}"
    )
    return send_slack(text)


def alert_drift_detected(report_path: str, summary: str = "") -> bool:
    """Alert on data drift detection.
    
    Args:
        report_path: Path to drift report
        summary: Optional summary of drift findings
        
    Returns:
        True if alert sent successfully
    """
    text = f":rotating_light: Data drift detected. Report: {report_path}\n{summary}"
    return send_slack(text)


def alert_promotion_success(model_name: str, version: str, metric_improvement: str = "") -> bool:
    """Alert on successful model promotion.
    
    Args:
        model_name: Name of promoted model
        version: Version number promoted to production
        metric_improvement: Optional improvement summary
        
    Returns:
        True if alert sent successfully
    """
    text = (
        f":white_check_mark: Model *{model_name}* v{version} promoted to Production. "
        f"{metric_improvement}"
    )
    return send_slack(text)


def alert_promotion_blocked(model_name: str, reason: str) -> bool:
    """Alert when promotion is blocked.
    
    Args:
        model_name: Name of model blocked from promotion
        reason: Reason for blocking
        
    Returns:
        True if alert sent successfully
    """
    text = f":no_entry: Model *{model_name}* promotion blocked: {reason}"
    return send_slack(text)
