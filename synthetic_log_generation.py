import random
import uuid
import json
from datetime import datetime, timedelta
import os

class WorkflowLogGenerator:
    def __init__(self):
        self.applications = [
            "AuthService", "PaymentGateway", "FraudDetection", 
            "UserManagement", "NotificationService", "AuditService",
            "BillingService", "RefundService", "ReportingService", "DatabaseService"
        ]

        self.payment_workflows = {
            "one_time_payment": {
                "required_apps": ["AuthService", "PaymentGateway", "DatabaseService"],
                "optional_apps": ["FraudDetection", "NotificationService", "AuditService"],
                "description": "Single payment transaction",
                "process_prefix": "onetimepay",
                "workflow_sequence": ["AuthService", "FraudDetection", "PaymentGateway", "DatabaseService", "NotificationService", "AuditService"]
            },
            "recurring_payment": {
                "required_apps": ["AuthService", "BillingService", "PaymentGateway", "DatabaseService"],
                "optional_apps": ["FraudDetection", "NotificationService", "AuditService"],
                "description": "Scheduled recurring payment",
                "process_prefix": "recurpay",
                "workflow_sequence": ["AuthService", "BillingService", "FraudDetection", "PaymentGateway", "DatabaseService", "NotificationService", "AuditService"]
            },
            "auto_pay_setup": {
                "required_apps": ["AuthService", "UserManagement", "BillingService", "DatabaseService"],
                "optional_apps": ["NotificationService", "AuditService"],
                "description": "Setting up automatic payment",
                "process_prefix": "autopaysetup",
                "workflow_sequence": ["AuthService", "UserManagement", "BillingService", "DatabaseService", "NotificationService", "AuditService"]
            },
            "refund_processing": {
                "required_apps": ["AuthService", "RefundService", "PaymentGateway", "DatabaseService"],
                "optional_apps": ["NotificationService", "AuditService", "ReportingService"],
                "description": "Processing payment refund",
                "process_prefix": "refundproc",
                "workflow_sequence": ["AuthService", "RefundService", "PaymentGateway", "DatabaseService", "NotificationService", "AuditService", "ReportingService"]
            },
            "payment_verification": {
                "required_apps": ["AuthService", "FraudDetection", "PaymentGateway"],
                "optional_apps": ["NotificationService", "AuditService", "DatabaseService"],
                "description": "Verifying suspicious payment",
                "process_prefix": "payverify",
                "workflow_sequence": ["AuthService", "FraudDetection", "PaymentGateway", "DatabaseService", "NotificationService", "AuditService"]
            },
            "failed_payment_retry": {
                "required_apps": ["AuthService", "PaymentGateway", "BillingService"],
                "optional_apps": ["NotificationService", "UserManagement", "DatabaseService"],
                "description": "Retrying failed payment",
                "process_prefix": "payretry",
                "workflow_sequence": ["AuthService", "BillingService", "PaymentGateway", "UserManagement", "DatabaseService", "NotificationService"]
            },
            "bulk_payment_processing": {
                "required_apps": ["AuthService", "BillingService", "PaymentGateway", "DatabaseService", "ReportingService"],
                "optional_apps": ["NotificationService", "AuditService"],
                "description": "Processing multiple payments in batch",
                "process_prefix": "bulkpay",
                "workflow_sequence": ["AuthService", "BillingService", "PaymentGateway", "DatabaseService", "ReportingService", "NotificationService", "AuditService"]
            }
        }
        
        # Application-specific message patterns
        self.app_messages = {
            "AuthService": {
                "success": ["User authenticated successfully", "Token validated", "Session created", "Authorization granted"],
                "error": ["Authentication failed - invalid credentials", "Token expired", "Session timeout", "Access denied"],
                "warning": ["Multiple login attempts detected", "Token expiring soon", "Unusual login location"],
                "debug": ["Validating user credentials", "Generating session token", "Checking user permissions"]
            },
            "PaymentGateway": {
                "success": ["Payment processed successfully", "Transaction approved", "Card charged successfully", "Payment confirmed"],
                "error": ["Payment declined - insufficient funds", "Card processing error", "Gateway timeout", "Invalid payment method"],
                "warning": ["High transaction amount flagged", "Unusual spending pattern", "Payment method expires soon"],
                "debug": ["Connecting to payment processor", "Validating card details", "Processing payment request"]
            },
            "FraudDetection": {
                "success": ["Transaction cleared by fraud check", "Risk assessment passed", "Fraud score within limits"],
                "error": ["High fraud risk detected", "Suspicious activity blocked", "Fraud rules triggered"],
                "warning": ["Medium fraud risk detected", "Unusual transaction pattern", "Location mismatch detected"],
                "debug": ["Running fraud detection algorithms", "Calculating risk score", "Checking transaction history"]
            },
            "BillingService": {
                "success": ["Invoice generated successfully", "Billing cycle processed", "Payment scheduled", "Account balance updated"],
                "error": ["Billing calculation failed", "Invoice generation error", "Account balance mismatch"],
                "warning": ["Payment overdue", "Account balance low", "Billing anomaly detected"],
                "debug": ["Calculating billing amount", "Processing billing cycle", "Updating account records"]
            },
            "NotificationService": {
                "success": ["Email notification sent", "SMS delivered successfully", "Push notification sent"],
                "error": ["Email delivery failed", "SMS service unavailable", "Notification template error"],
                "warning": ["High notification volume", "Delivery delay detected", "Template rendering slow"],
                "debug": ["Preparing notification content", "Selecting delivery channel", "Queuing notification"]
            },
            "UserManagement": {
                "success": ["User profile updated", "Account settings saved", "User preferences applied"],
                "error": ["User not found", "Profile update failed", "Invalid user data"],
                "warning": ["Account requires verification", "Profile incomplete", "Multiple accounts detected"],
                "debug": ["Loading user profile", "Validating user data", "Updating user records"]
            },
            "AuditService": {
                "success": ["Audit log created", "Compliance check passed", "Activity recorded"],
                "error": ["Audit logging failed", "Compliance violation detected", "Log storage error"],
                "warning": ["Audit log volume high", "Compliance check delayed", "Storage capacity warning"],
                "debug": ["Recording audit event", "Validating compliance rules", "Archiving audit logs"]
            },
            "RefundService": {
                "success": ["Refund processed successfully", "Refund approved", "Amount credited to account"],
                "error": ["Refund processing failed", "Refund policy violation", "Insufficient refund balance"],
                "warning": ["Large refund amount flagged", "Multiple refund requests", "Refund deadline approaching"],
                "debug": ["Validating refund eligibility", "Calculating refund amount", "Processing refund request"]
            },
            "ReportingService": {
                "success": ["Report generated successfully", "Data exported", "Analytics updated"],
                "error": ["Report generation failed", "Data export error", "Analytics calculation error"],
                "warning": ["Large dataset processing", "Report generation slow", "Data quality issues"],
                "debug": ["Aggregating transaction data", "Applying report filters", "Formatting report output"]
            },
            "DatabaseService": {
                "success": ["Data saved successfully", "Transaction committed", "Database updated"],
                "error": ["Database connection failed", "Transaction rolled back", "Data integrity violation"],
                "warning": ["Slow query detected", "Connection pool exhausted", "Database performance degraded"],
                "debug": ["Executing database query", "Starting transaction", "Validating data constraints"]
            }
        }
    
    def generate_correlation_id(self):
        return f"CORR-{uuid.uuid4().hex[:12].upper()}"
    
    def generate_process_name(self, workflow_type, app_name):
        """Generate consistent process name with format: processprefix_applicationname"""
        process_prefix = self.payment_workflows[workflow_type]["process_prefix"]
        return f"{process_prefix}_{app_name}".lower()
    
    def get_consistent_workflow_sequence(self, workflow_type):
        """Get the consistent workflow sequence for a given workflow type"""
        workflow_config = self.payment_workflows[workflow_type]
        workflow_sequence = workflow_config["workflow_sequence"].copy()
        
        # Use workflow_type as seed for consistency
        random.seed(hash(workflow_type) % 1000)
        
        required_apps = set(workflow_config["required_apps"])
        optional_apps = set(workflow_config["optional_apps"])
        
        # Filter the sequence to include required apps and randomly selected optional apps
        participating_apps = []
        for app in workflow_sequence:
            if app in required_apps:
                participating_apps.append(app)
            elif app in optional_apps and random.random() < 0.7:  
                participating_apps.append(app)
        
        random.seed()
        
        return participating_apps
    
    def generate_log_entry(self, correlation_id, app_name, timestamp, log_level="INFO", workflow_type=""):
        process_name = self.generate_process_name(workflow_type, app_name)
        
        messages = self.app_messages.get(app_name, {
            "success": ["Operation completed"],
            "error": ["Operation failed"],
            "warning": ["Warning occurred"],
            "debug": ["Debug information"]
        })
        
        if log_level == "ERROR":
            message = random.choice(messages.get("error", ["Operation failed"]))
            details = f"Error in {app_name}: {message} | Process: {process_name} | StackTrace: com.payment.{app_name.lower()}.Exception"
        elif log_level == "WARN":
            message = random.choice(messages.get("warning", ["Warning occurred"]))
            details = f"Warning in {app_name}: {message} | Process: {process_name} | Action: Monitor closely"
        elif log_level == "DEBUG":
            message = random.choice(messages.get("debug", ["Debug information"]))
            details = f"Debug {app_name}: {message} | Process: {process_name} | Method: process_{random.choice(['request', 'data', 'transaction'])}"
        else:
            message = random.choice(messages.get("success", ["Operation completed"]))
            details = f"{app_name}: {message} | Process: {process_name} | Duration: {random.randint(10, 2000)}ms"
        
        return {
            "timestamp": timestamp.isoformat(),
            "correlation_id": correlation_id,
            "application": app_name,
            "process_name": process_name,  
            "log_level": log_level,
            "message": message,
            "details": details,
            "user_id": f"user_{random.randint(100, 999)}",
            "session_id": f"session_{uuid.uuid4().hex[:8]}",
            "request_id": f"req_{uuid.uuid4().hex[:6]}",
            "host": f"payment-server-{random.randint(1, 5)}.company.com"
        }
    
    def generate_workflow_logs(self, correlation_id, workflow_type):
        logs = []
        base_time = datetime.now() - timedelta(days=random.randint(0, 30))
        
        participating_apps = self.get_consistent_workflow_sequence(workflow_type)
        anomaly_type = self.determine_anomaly()
        
        if anomaly_type == "missing_required_app":
            workflow_config = self.payment_workflows[workflow_type]
            required_apps = workflow_config["required_apps"]
            if len([app for app in participating_apps if app in required_apps]) > 1:
                required_participating = [app for app in participating_apps if app in required_apps[1:]]
                if required_participating:
                    missing_app = random.choice(required_participating)
                    participating_apps.remove(missing_app)
                    print(f"Anomaly: Missing required app {missing_app} for correlation {correlation_id}")
        
        elif anomaly_type == "early_failure":
            error_at_step = random.randint(0, min(2, len(participating_apps) - 1))
        elif anomaly_type == "timeout":
            timeout_count = min(2, len(participating_apps))
            timeout_apps = random.sample(participating_apps, timeout_count)
            for app in timeout_apps:
                participating_apps.remove(app)
        elif anomaly_type == "partial_failure":
            error_at_step = random.randint(len(participating_apps) // 2, len(participating_apps) - 1)
        else:
            # Normal flow
            error_at_step = -1
        
        # Generate logs for participating applications in consistent order
        for i, app in enumerate(participating_apps):
            step_time = base_time + timedelta(seconds=i * random.randint(1, 8))
            
            # Determine log level based on anomaly and position
            if anomaly_type in ["early_failure", "partial_failure"] and i == error_at_step:
                log_level = "ERROR"
            elif random.random() < 0.15:  # 15% chance of warning
                log_level = "WARN"
            else:
                log_level = "INFO"
            
            log_entry = self.generate_log_entry(correlation_id, app, step_time, log_level, workflow_type)
            logs.append(log_entry)
            
            # Add debug logs occasionally
            if random.random() < 0.4:  # 40% chance of debug log
                debug_time = step_time + timedelta(milliseconds=random.randint(10, 500))
                debug_log = self.generate_log_entry(correlation_id, app, debug_time, "DEBUG", workflow_type)
                debug_log["message"] = f"Processing {workflow_type} step {i+1}"
                logs.append(debug_log)
            
            # Stop generating logs if error occurred and it's a critical failure
            if log_level == "ERROR" and anomaly_type == "early_failure":
                break
        
        return logs, anomaly_type
    
    def determine_anomaly(self):
        """Determine what type of anomaly (if any) should occur"""
        anomaly_chance = random.random()
        
        if anomaly_chance < 0.05:  # 5% chance
            return "missing_required_app"
        elif anomaly_chance < 0.10:  # 5% chance
            return "early_failure"
        elif anomaly_chance < 0.15:  # 5% chance
            return "timeout"
        elif anomaly_chance < 0.25:  # 10% chance
            return "partial_failure"
        else:
            return "normal"  # 75% chance of normal flow
    
    def generate_payment_dataset(self, num_workflows=200):
        all_logs = []
        correlation_metadata = []
        
        workflow_types = list(self.payment_workflows.keys())
        
        print("Generating payment workflow dataset with consistent workflows and process names...")
        
        for i in range(num_workflows):
            correlation_id = self.generate_correlation_id()
            workflow_type = random.choice(workflow_types)
            
            workflow_logs, anomaly_type = self.generate_workflow_logs(correlation_id, workflow_type)
            all_logs.extend(workflow_logs)
            
            # Track metadata for analysis
            correlation_metadata.append({
                "correlation_id": correlation_id,
                "workflow_type": workflow_type,
                "anomaly_type": anomaly_type,
                "log_count": len(workflow_logs),
                "participating_apps": list(set([log["application"] for log in workflow_logs])),
                "process_names": list(set([log["process_name"] for log in workflow_logs])),
                "has_error": any(log["log_level"] == "ERROR" for log in workflow_logs)
            })
            
            if i % 50 == 0:
                print(f"Generated {i} workflows...")
        
        return all_logs, correlation_metadata
    
    def save_dataset(self, logs, metadata, output_dir="synthetic_logs"):
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all logs to a single file
        with open(f"{output_dir}/all_payment_logs.jsonl", 'w') as f:
            for log in logs:
                f.write(json.dumps(log) + '\n')
        
        # Save logs by application
        app_logs = {}
        for log in logs:
            app = log["application"]
            if app not in app_logs:
                app_logs[app] = []
            app_logs[app].append(log)
        
        for app, logs_list in app_logs.items():
            filename = f"{output_dir}/{app}_logs.jsonl"
            with open(filename, 'w') as f:
                for log in logs_list:
                    f.write(json.dumps(log) + '\n')
        

        
        # Save metadata for analysis
        with open(f"{output_dir}/workflow_metadata.jsonl", 'w') as f:
            for meta in metadata:
                f.write(json.dumps(meta) + '\n')
        
        # Save correlation IDs
        correlation_ids = [meta["correlation_id"] for meta in metadata]
        with open(f"{output_dir}/correlation_ids.txt", 'w') as f:
            for corr_id in correlation_ids:
                f.write(corr_id + '\n')
        
        return app_logs
    
    def generate_analysis_report(self, metadata):
        print("\n" + "="*60)
        print("PAYMENT WORKFLOW ANALYSIS REPORT")
        print("="*60)
        
        total_workflows = len(metadata)
        
        # Workflow type distribution
        workflow_counts = {}
        for meta in metadata:
            wf_type = meta["workflow_type"]
            workflow_counts[wf_type] = workflow_counts.get(wf_type, 0) + 1
        
        print(f"\nWorkflow Type Distribution:")
        for wf_type, count in sorted(workflow_counts.items()):
            percentage = (count / total_workflows) * 100
            print(f"  {wf_type}: {count} ({percentage:.1f}%)")
        
        # Process name analysis
        all_process_names = set()
        for meta in metadata:
            all_process_names.update(meta["process_names"])
        
        print(f"\nUnique Process Names Generated: {len(all_process_names)}")
        print("Sample process names:")
        for i, process_name in enumerate(sorted(list(all_process_names))[:10]):
            print(f"  {process_name}")
        if len(all_process_names) > 10:
            print(f"  ... and {len(all_process_names) - 10} more")
        
        # Workflow consistency check
        print(f"\nWorkflow Consistency Check:")
        workflow_sequences = {}
        for wf_type in self.payment_workflows.keys():
            consistent_sequence = self.get_consistent_workflow_sequence(wf_type)
            workflow_sequences[wf_type] = consistent_sequence
            print(f"  {wf_type}: {' -> '.join(consistent_sequence)}")
        
        # Anomaly distribution
        anomaly_counts = {}
        for meta in metadata:
            anomaly = meta["anomaly_type"]
            anomaly_counts[anomaly] = anomaly_counts.get(anomaly, 0) + 1
        
        print(f"\nAnomaly Distribution:")
        for anomaly, count in sorted(anomaly_counts.items()):
            percentage = (count / total_workflows) * 100
            print(f"  {anomaly}: {count} ({percentage:.1f}%)")
        
        # Application participation analysis
        app_participation = {}
        for app in self.applications:
            participation_count = 0
            for meta in metadata:
                if app in meta["participating_apps"]:
                    participation_count += 1
            app_participation[app] = participation_count
        
        print(f"\nApplication Participation (out of {total_workflows} workflows):")
        for app, count in sorted(app_participation.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_workflows) * 100
            print(f"  {app}: {count} ({percentage:.1f}%)")
        
        # Error analysis
        error_workflows = [meta for meta in metadata if meta["has_error"]]
        print(f"\nError Analysis:")
        print(f"  Workflows with errors: {len(error_workflows)} ({len(error_workflows)/total_workflows*100:.1f}%)")
        
        return {
            "total_workflows": total_workflows,
            "workflow_distribution": workflow_counts,
            "anomaly_distribution": anomaly_counts,
            "app_participation": app_participation,
            "process_names": list(all_process_names),
            "workflow_sequences": workflow_sequences,
            "error_rate": len(error_workflows) / total_workflows
        }

def main():
    generator = WorkflowLogGenerator()
    
    # Generate dataset
    logs, metadata = generator.generate_payment_dataset(num_workflows=300)
    
    print(f"\nGenerated {len(logs)} total log entries for {len(metadata)} payment workflows")
    
    # Save dataset
    app_logs = generator.save_dataset(logs, metadata)

    analysis = generator.generate_analysis_report(metadata)
    
    print(f"\nFiles saved in 'synthetic_logs' directory:")
    print(f"  - all_payment_logs.jsonl: All logs combined")
    print(f"  - workflow_metadata.jsonl: Workflow metadata for analysis")
    print(f"  - correlation_ids.txt: List of all correlation IDs")
    print(f"  - Individual application log files")
    
    print(f"\nDataset Statistics:")
    print(f"  Total logs: {len(logs)}")
    print(f"  Total workflows: {len(metadata)}")
    print(f"  Average logs per workflow: {len(logs) / len(metadata):.1f}")
    print(f"  Applications with logs: {len(app_logs)}")
    
    # Show sample logs with process names
    print(f"\nSample log entries with process names:")
    for i, log in enumerate(logs[:5]):
        print(f"{i+1}. Process: {log['process_name']}, App: {log['application']}, "
              f"Level: {log['log_level']}, Message: {log['message'][:50]}...")

if __name__ == "__main__":
    main()