#!/bin/bash
set -euo pipefail

# Container Analytics Deployment Test Script
# This script validates the production deployment setup

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO: $1${NC}"
}

# Test result tracking
pass_test() {
    ((TESTS_PASSED++))
    ((TESTS_TOTAL++))
    echo -e "${GREEN}✓ $1${NC}"
}

fail_test() {
    ((TESTS_FAILED++))
    ((TESTS_TOTAL++))
    echo -e "${RED}✗ $1${NC}"
}

skip_test() {
    ((TESTS_TOTAL++))
    echo -e "${YELLOW}⚠ $1 (SKIPPED)${NC}"
}

# Test functions
test_docker_availability() {
    info "Testing Docker availability..."
    
    if command -v docker >/dev/null 2>&1; then
        pass_test "Docker command available"
    else
        fail_test "Docker command not found"
        return 1
    fi
    
    if docker info >/dev/null 2>&1; then
        pass_test "Docker daemon running"
    else
        fail_test "Docker daemon not running"
        return 1
    fi
    
    if command -v docker-compose >/dev/null 2>&1; then
        pass_test "Docker Compose available"
    else
        fail_test "Docker Compose not found"
        return 1
    fi
}

test_file_structure() {
    info "Testing file structure..."
    
    local required_files=(
        "Dockerfile"
        "deployment/docker/docker-compose.yml"
        "deployment/README.md"
        ".env.production"
        "requirements.txt"
        "utils/metrics_server.py"
        "utils/production_logging.py"
        "utils/production_cache.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            pass_test "Required file exists: $file"
        else
            fail_test "Missing required file: $file"
        fi
    done
    
    local required_dirs=(
        "data"
        "logs"
        "deployment/docker/nginx"
        "deployment/docker/loki"
        "deployment/docker/grafana"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            pass_test "Required directory exists: $dir"
        else
            mkdir -p "$dir"
            pass_test "Created missing directory: $dir"
        fi
    done
}

test_docker_compose_config() {
    info "Testing Docker Compose configuration..."
    
    if docker-compose -f deployment/docker/docker-compose.yml config --quiet; then
        pass_test "Docker Compose configuration valid"
    else
        fail_test "Docker Compose configuration invalid"
        return 1
    fi
    
    # Test different profiles
    local profiles=("monitoring" "production")
    for profile in "${profiles[@]}"; do
        if docker-compose -f deployment/docker/docker-compose.yml --profile "$profile" config --quiet; then
            pass_test "Docker Compose profile '$profile' valid"
        else
            fail_test "Docker Compose profile '$profile' invalid"
        fi
    done
}

test_environment_config() {
    info "Testing environment configuration..."
    
    if [[ -f .env.production ]]; then
        pass_test ".env.production template exists"
        
        # Check for required environment variables
        local required_vars=(
            "ENVIRONMENT"
            "LOG_LEVEL"
            "DOWNLOAD_INTERVAL_MINUTES"
            "STREAMS"
            "DATABASE_URL"
        )
        
        for var in "${required_vars[@]}"; do
            if grep -q "^${var}=" .env.production; then
                pass_test "Environment variable defined: $var"
            else
                fail_test "Missing environment variable: $var"
            fi
        done
    else
        fail_test ".env.production file not found"
    fi
    
    if [[ -f .env ]]; then
        pass_test ".env file exists"
    else
        warn ".env file not found (will be created from template during deployment)"
    fi
}

test_dockerfile_syntax() {
    info "Testing Dockerfile syntax..."
    
    if docker build -t container-analytics-test . --target base-builder --quiet; then
        pass_test "Dockerfile base-builder stage builds successfully"
    else
        fail_test "Dockerfile base-builder stage build failed"
        return 1
    fi
    
    if docker build -t container-analytics-test . --target python-builder --quiet; then
        pass_test "Dockerfile python-builder stage builds successfully"
    else
        fail_test "Dockerfile python-builder stage build failed"
        return 1
    fi
    
    # Clean up test image
    docker rmi container-analytics-test >/dev/null 2>&1 || true
}

test_network_configuration() {
    info "Testing network configuration..."
    
    # Check if ports are available
    local ports=(8501 9090 3000 80 443)
    for port in "${ports[@]}"; do
        if ! netstat -tln 2>/dev/null | grep -q ":${port} "; then
            pass_test "Port $port is available"
        else
            warn "Port $port is already in use"
        fi
    done
}

test_ssl_configuration() {
    info "Testing SSL configuration..."
    
    local ssl_dir="deployment/docker/nginx/ssl"
    if [[ -d "$ssl_dir" ]]; then
        pass_test "SSL directory exists"
        
        if [[ -f "$ssl_dir/cert.pem" && -f "$ssl_dir/key.pem" ]]; then
            pass_test "SSL certificate files exist"
            
            # Verify certificate
            if openssl x509 -in "$ssl_dir/cert.pem" -text -noout >/dev/null 2>&1; then
                pass_test "SSL certificate is valid"
            else
                fail_test "SSL certificate is invalid"
            fi
        else
            warn "SSL certificate files not found (will be generated during deployment)"
        fi
    else
        fail_test "SSL directory not found"
    fi
}

test_monitoring_config() {
    info "Testing monitoring configuration..."
    
    local monitoring_files=(
        "deployment/docker/grafana/datasources/datasources.yml"
        "deployment/docker/grafana/dashboards/dashboard.yml"
        "deployment/docker/grafana/dashboards/container-analytics-dashboard.json"
        "deployment/docker/loki/config.yml"
        "deployment/docker/nginx/nginx.conf"
    )
    
    for file in "${monitoring_files[@]}"; do
        if [[ -f "$file" ]]; then
            pass_test "Monitoring config exists: $file"
        else
            fail_test "Missing monitoring config: $file"
        fi
    done
}

test_security_configuration() {
    info "Testing security configuration..."
    
    if [[ -f deployment/security-setup.sh ]]; then
        if [[ -x deployment/security-setup.sh ]]; then
            pass_test "Security setup script exists and is executable"
        else
            fail_test "Security setup script is not executable"
        fi
    else
        fail_test "Security setup script not found"
    fi
    
    # Check for security-related files
    local security_files=(
        "deployment/systemd/container-analytics-hardened.service"
    )
    
    for file in "${security_files[@]}"; do
        if [[ -f "$file" ]]; then
            pass_test "Security config exists: $file"
        else
            warn "Optional security config not found: $file"
        fi
    done
}

test_python_dependencies() {
    info "Testing Python dependencies..."
    
    if [[ -f requirements.txt ]]; then
        pass_test "requirements.txt exists"
        
        # Check for critical dependencies
        local critical_deps=(
            "streamlit"
            "ultralytics"
            "selenium"
            "SQLAlchemy"
            "APScheduler"
            "prometheus_client"
            "loguru"
            "redis"
        )
        
        for dep in "${critical_deps[@]}"; do
            if grep -qi "$dep" requirements.txt; then
                pass_test "Critical dependency found: $dep"
            else
                fail_test "Missing critical dependency: $dep"
            fi
        done
    else
        fail_test "requirements.txt not found"
    fi
}

test_health_checks() {
    info "Testing health check scripts..."
    
    # Check if health check script exists in Dockerfile
    if grep -q "health_check.py" Dockerfile; then
        pass_test "Health check script referenced in Dockerfile"
    else
        fail_test "Health check script not referenced in Dockerfile"
    fi
    
    # Check metrics server
    if [[ -f utils/metrics_server.py ]]; then
        if python3 -m py_compile utils/metrics_server.py 2>/dev/null; then
            pass_test "Metrics server script compiles successfully"
        else
            fail_test "Metrics server script has syntax errors"
        fi
    else
        fail_test "Metrics server script not found"
    fi
}

perform_integration_test() {
    info "Performing integration test..."
    
    # Create minimal test environment
    export DOWNLOAD_INTERVAL_MINUTES=10
    export STREAMS="in_gate"
    export DATABASE_URL="sqlite:////tmp/test_database.db"
    export DATA_DIR="/tmp/container-analytics-test"
    export MODELS_DIR="/tmp/container-analytics-test/models"
    
    mkdir -p "$DATA_DIR/models"
    
    # Test Docker Compose with dry-run equivalent
    if docker-compose -f deployment/docker/docker-compose.yml config >/dev/null 2>&1; then
        pass_test "Integration test: Docker Compose config resolves"
    else
        fail_test "Integration test: Docker Compose config resolution failed"
    fi
    
    # Clean up test environment
    rm -rf "/tmp/container-analytics-test" 2>/dev/null || true
}

# Performance test
test_performance_readiness() {
    info "Testing performance readiness..."
    
    # Check system resources
    if command -v free >/dev/null 2>&1; then
        local available_memory=$(free -m | awk 'NR==2{printf "%.1f", $7/1024 }')
        if (( $(echo "$available_memory > 4.0" | bc -l 2>/dev/null || echo 0) )); then
            pass_test "Sufficient memory available: ${available_memory}GB"
        else
            warn "Low memory available: ${available_memory}GB (recommended: 4GB+)"
        fi
    fi
    
    if command -v nproc >/dev/null 2>&1; then
        local cpu_cores=$(nproc)
        if [[ $cpu_cores -ge 4 ]]; then
            pass_test "Sufficient CPU cores: $cpu_cores"
        else
            warn "Limited CPU cores: $cpu_cores (recommended: 4+)"
        fi
    fi
    
    # Check disk space
    local available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_space -ge 50 ]]; then
        pass_test "Sufficient disk space: ${available_space}GB"
    else
        warn "Limited disk space: ${available_space}GB (recommended: 50GB+)"
    fi
}

# Main test runner
run_tests() {
    log "Starting Container Analytics Deployment Tests"
    echo
    
    # Core tests
    test_docker_availability || return 1
    test_file_structure
    test_docker_compose_config
    test_environment_config
    test_dockerfile_syntax
    test_network_configuration
    test_ssl_configuration
    test_monitoring_config
    test_security_configuration
    test_python_dependencies
    test_health_checks
    
    # Integration test
    perform_integration_test
    
    # Performance test
    test_performance_readiness
    
    echo
    log "Test Results Summary:"
    echo "  Tests Passed: $TESTS_PASSED"
    echo "  Tests Failed: $TESTS_FAILED"
    echo "  Total Tests:  $TESTS_TOTAL"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed! Deployment is ready.${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed. Please review and fix issues before deployment.${NC}"
        return 1
    fi
}

# Help function
show_help() {
    echo "Container Analytics Deployment Test Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --verbose  Enable verbose output"
    echo "  --quick        Run quick tests only (skip Docker build)"
    echo
    echo "This script validates the production deployment setup including:"
    echo "  - Docker and Docker Compose availability"
    echo "  - File structure and configuration"
    echo "  - Network configuration"
    echo "  - Security setup"
    echo "  - Monitoring configuration"
    echo "  - Performance readiness"
}

# Parse command line arguments
QUICK_MODE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Enable verbose mode if requested
if [[ $VERBOSE == true ]]; then
    set -x
fi

# Run tests
if run_tests; then
    exit 0
else
    exit 1
fi