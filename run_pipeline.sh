#!/bin/bash

# MLOps Pipeline Runner & Monitor
# This script helps you trigger and monitor the MLOps pipeline

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/Users/yashvardhanjain/Downloads/mlops_project_clean"
AIRFLOW_CONTAINER="mlops_project_clean-airflow-1"
DAG_NAME="mlops_full_pipeline"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         MLOps Pipeline Runner & Monitor               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check if services are running
check_services() {
    echo -e "${YELLOW}[1/6] Checking Docker services...${NC}"
    
    if ! docker-compose ps | grep -q "Up"; then
        echo -e "${RED}❌ Docker services are not running!${NC}"
        echo "Please start services with: docker-compose up -d"
        exit 1
    fi
    
    # Check each service
    services=("airflow" "mlflow" "postgres" "minio" "redis")
    for service in "${services[@]}"; do
        if docker-compose ps | grep "$service" | grep -q "Up"; then
            echo -e "${GREEN}   ✓ $service is running${NC}"
        else
            echo -e "${RED}   ✗ $service is NOT running${NC}"
        fi
    done
    echo ""
}

# Function to check data file
check_data() {
    echo -e "${YELLOW}[2/6] Checking data file...${NC}"
    
    if [ -f "$PROJECT_DIR/data/raw/data.csv" ]; then
        line_count=$(wc -l < "$PROJECT_DIR/data/raw/data.csv")
        echo -e "${GREEN}   ✓ data.csv exists ($line_count lines)${NC}"
        echo "   Preview:"
        head -3 "$PROJECT_DIR/data/raw/data.csv" | sed 's/^/     /'
    else
        echo -e "${RED}   ✗ data.csv NOT found!${NC}"
        echo "   Creating from sample_data.csv..."
        cp "$PROJECT_DIR/data/raw/sample_data.csv" "$PROJECT_DIR/data/raw/data.csv"
        echo -e "${GREEN}   ✓ Created data.csv${NC}"
    fi
    echo ""
}

# Function to trigger DAG
trigger_dag() {
    echo -e "${YELLOW}[3/6] Triggering DAG...${NC}"
    
    # Trigger the DAG
    docker exec "$AIRFLOW_CONTAINER" airflow dags trigger "$DAG_NAME" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}   ✓ DAG triggered successfully!${NC}"
        echo "   DAG: $DAG_NAME"
        echo "   Time: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo -e "${RED}   ✗ Failed to trigger DAG${NC}"
        exit 1
    fi
    echo ""
}

# Function to monitor DAG execution
monitor_dag() {
    echo -e "${YELLOW}[4/6] Monitoring DAG execution...${NC}"
    echo "   (This may take 2-3 minutes)"
    echo ""
    
    # Wait a moment for DAG to start
    sleep 3
    
    # Monitor for up to 5 minutes
    max_wait=300
    elapsed=0
    
    while [ $elapsed -lt $max_wait ]; do
        # Get DAG state
        state=$(docker exec "$AIRFLOW_CONTAINER" \
                airflow dags state "$DAG_NAME" $(date +%Y-%m-%d) 2>/dev/null | tail -1)
        
        if [[ "$state" == *"success"* ]]; then
            echo -e "${GREEN}   ✓ DAG completed successfully!${NC}"
            return 0
        elif [[ "$state" == *"failed"* ]]; then
            echo -e "${RED}   ✗ DAG failed!${NC}"
            return 1
        elif [[ "$state" == *"running"* ]]; then
            echo -ne "   ⏳ Running... (${elapsed}s elapsed)\r"
        fi
        
        sleep 5
        elapsed=$((elapsed + 5))
    done
    
    echo -e "${YELLOW}   ⚠ Timeout reached. Check Airflow UI for status.${NC}"
    return 2
}

# Function to check outputs
check_outputs() {
    echo -e "${YELLOW}[5/6] Checking pipeline outputs...${NC}"
    
    expected_files=(
        "data/clean/data.csv:Cleaned data"
        "data/features/data.csv:Feature-engineered data"
        "data/train.csv:Training split"
        "data/validation.csv:Validation split"
        "data/monitoring/drift_report.html:Drift report"
        "data/model_card.html:Model card"
    )
    
    all_exist=true
    for entry in "${expected_files[@]}"; do
        IFS=":" read -r file desc <<< "$entry"
        full_path="$PROJECT_DIR/$file"
        
        if [ -f "$full_path" ]; then
            size=$(ls -lh "$full_path" | awk '{print $5}')
            echo -e "${GREEN}   ✓ $desc ($size)${NC}"
        else
            echo -e "${RED}   ✗ $desc NOT found${NC}"
            all_exist=false
        fi
    done
    
    if [ "$all_exist" = true ]; then
        echo -e "${GREEN}   ✓ All expected outputs present!${NC}"
    fi
    echo ""
}

# Function to display results
show_results() {
    echo -e "${YELLOW}[6/6] Pipeline Results Summary${NC}"
    echo ""
    
    # Show MLflow info
    echo -e "${BLUE}📊 MLflow Experiment:${NC}"
    echo "   URL: http://localhost:5050"
    echo "   Experiment: demo_experiment"
    echo ""
    
    # Show Airflow info
    echo -e "${BLUE}🔄 Airflow DAG:${NC}"
    echo "   URL: http://localhost:8099"
    echo "   DAG: $DAG_NAME"
    echo ""
    
    # Show generated reports
    echo -e "${BLUE}📄 Generated Reports:${NC}"
    
    if [ -f "$PROJECT_DIR/data/monitoring/drift_report.html" ]; then
        echo "   - Drift Report: $PROJECT_DIR/data/monitoring/drift_report.html"
    fi
    
    if [ -f "$PROJECT_DIR/data/model_card.html" ]; then
        echo "   - Model Card: $PROJECT_DIR/data/model_card.html"
    fi
    echo ""
    
    # Show commands to open reports
    echo -e "${BLUE}💡 Quick Commands:${NC}"
    echo "   # Open Drift Report"
    echo "   open $PROJECT_DIR/data/monitoring/drift_report.html"
    echo ""
    echo "   # Open Model Card"
    echo "   open $PROJECT_DIR/data/model_card.html"
    echo ""
    echo "   # View Airflow UI"
    echo "   open http://localhost:8099"
    echo ""
    echo "   # View MLflow UI"
    echo "   open http://localhost:5050"
    echo ""
}

# Main execution
main() {
    # Parse command line arguments
    MONITOR=true
    if [ "$1" = "--no-monitor" ]; then
        MONITOR=false
    fi
    
    cd "$PROJECT_DIR"
    
    check_services
    check_data
    trigger_dag
    
    if [ "$MONITOR" = true ]; then
        monitor_dag
        dag_result=$?
        
        echo ""
        check_outputs
        show_results
        
        if [ $dag_result -eq 0 ]; then
            echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
            echo -e "${GREEN}║  🎉 Pipeline completed successfully!                  ║${NC}"
            echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
            
            # Ask if user wants to open reports
            echo ""
            read -p "Open drift report in browser? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                open "$PROJECT_DIR/data/monitoring/drift_report.html"
            fi
            
            read -p "Open model card in browser? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                open "$PROJECT_DIR/data/model_card.html"
            fi
            
        elif [ $dag_result -eq 1 ]; then
            echo -e "${RED}╔════════════════════════════════════════════════════════╗${NC}"
            echo -e "${RED}║  ❌ Pipeline failed! Check Airflow logs.              ║${NC}"
            echo -e "${RED}╚════════════════════════════════════════════════════════╝${NC}"
        else
            echo -e "${YELLOW}╔════════════════════════════════════════════════════════╗${NC}"
            echo -e "${YELLOW}║  ⏳ Pipeline is still running. Check Airflow UI.      ║${NC}"
            echo -e "${YELLOW}╚════════════════════════════════════════════════════════╝${NC}"
        fi
    else
        echo -e "${GREEN}✓ DAG triggered. Monitor progress in Airflow UI.${NC}"
        show_results
    fi
}

# Run main function
main "$@"
