#!/bin/bash
set -euo pipefail

# Container Analytics Production Security Setup Script
# This script implements security best practices for production deployment

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if running as root
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Generate secure passwords
generate_passwords() {
    log "Generating secure passwords..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        cp .env.production .env
        log "Created .env file from .env.production template"
    fi
    
    # Generate passwords
    DB_PASSWORD=$(openssl rand -base64 32)
    GRAFANA_PASSWORD=$(openssl rand -base64 32)
    SMTP_PASSWORD=$(openssl rand -base64 32)
    
    # Update .env file
    sed -i.bak "s/your-secure-db-password-here/$DB_PASSWORD/g" .env
    sed -i.bak "s/your-secure-grafana-password-here/$GRAFANA_PASSWORD/g" .env
    sed -i.bak "s/your-smtp-password/$SMTP_PASSWORD/g" .env
    
    # Remove backup file
    rm .env.bak
    
    log "Passwords generated and updated in .env file"
    warn "Please update SMTP_PASSWORD with your actual SMTP credentials"
}

# Set up SSL certificates
setup_ssl() {
    log "Setting up SSL certificates..."
    
    SSL_DIR="deployment/docker/nginx/ssl"
    mkdir -p "$SSL_DIR"
    
    if [[ ! -f "$SSL_DIR/cert.pem" || ! -f "$SSL_DIR/key.pem" ]]; then
        log "Generating self-signed SSL certificate for development..."
        
        # Generate self-signed certificate
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$SSL_DIR/key.pem" \
            -out "$SSL_DIR/cert.pem" \
            -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"
        
        warn "Self-signed certificate generated. Replace with CA-signed certificate for production"
    fi
    
    # Set proper permissions
    chmod 600 "$SSL_DIR"/*.pem
    
    log "SSL certificates configured"
}

# Configure file permissions
setup_file_permissions() {
    log "Configuring file permissions..."
    
    # Set proper permissions for configuration files
    chmod 600 .env
    chmod 644 .env.production
    chmod +x deployment/security-setup.sh
    
    # Set permissions for data directories
    if [[ -d data ]]; then
        chmod 755 data
        chmod -R 644 data/* 2>/dev/null || true
        find data -type d -exec chmod 755 {} \; 2>/dev/null || true
    fi
    
    # Set permissions for log directories
    if [[ -d logs ]]; then
        chmod 755 logs
        chmod 644 logs/* 2>/dev/null || true
    fi
    
    # Set permissions for deployment configs
    chmod -R 644 deployment/docker/
    find deployment/docker/ -type d -exec chmod 755 {} \;
    
    log "File permissions configured"
}

# Configure Docker security
setup_docker_security() {
    log "Configuring Docker security settings..."
    
    # Check if Docker is running in rootless mode
    if docker info 2>/dev/null | grep -q "rootless"; then
        log "Docker is running in rootless mode (good for security)"
    else
        warn "Docker is running in root mode. Consider rootless mode for better security"
    fi
    
    # Create Docker secrets directory
    mkdir -p deployment/docker/secrets
    chmod 700 deployment/docker/secrets
    
    # Generate secrets files for production
    echo "$DB_PASSWORD" > deployment/docker/secrets/db_password
    echo "$GRAFANA_PASSWORD" > deployment/docker/secrets/grafana_password
    
    chmod 600 deployment/docker/secrets/*
    
    log "Docker security configured"
}

# Set up firewall rules (if ufw is available)
setup_firewall() {
    if command -v ufw >/dev/null 2>&1; then
        log "Configuring UFW firewall rules..."
        
        # Allow SSH (be careful not to lock yourself out)
        sudo ufw allow ssh
        
        # Allow only necessary ports
        sudo ufw allow 80/tcp    # HTTP
        sudo ufw allow 443/tcp   # HTTPS
        
        # Optional: Allow specific monitoring ports (restrict to internal network)
        # sudo ufw allow from 10.0.0.0/8 to any port 3000  # Grafana
        # sudo ufw allow from 10.0.0.0/8 to any port 9090  # Metrics
        
        # Enable firewall (will prompt user)
        warn "About to enable UFW firewall. Make sure SSH access is configured!"
        read -p "Enable firewall? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo ufw --force enable
            log "Firewall enabled"
        else
            log "Firewall not enabled (user choice)"
        fi
    else
        warn "UFW not available. Please configure firewall manually"
    fi
}

# Configure log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    # Create logrotate configuration
    cat > deployment/logrotate-container-analytics << 'EOF'
/var/log/container-analytics/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    create 644 analytics analytics
}

/path/to/container-analytics/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF
    
    # Update path in logrotate config
    sed -i "s|/path/to/container-analytics|$(pwd)|g" deployment/logrotate-container-analytics
    
    warn "Logrotate config created at deployment/logrotate-container-analytics"
    warn "Install it with: sudo cp deployment/logrotate-container-analytics /etc/logrotate.d/"
    
    log "Log rotation configured"
}

# Create systemd hardening service
create_hardened_systemd_service() {
    log "Creating hardened systemd service..."
    
    cat > deployment/systemd/container-analytics-hardened.service << 'EOF'
[Unit]
Description=Container Analytics (Hardened)
After=docker.service
Requires=docker.service

[Service]
Type=forking
User=analytics
Group=analytics
WorkingDirectory=/opt/container-analytics

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/container-analytics/data
ReadWritePaths=/opt/container-analytics/logs
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictRealtime=true
RestrictNamespaces=true
RestrictSUIDSGID=true
LockPersonality=true
MemoryDenyWriteExecute=true
SystemCallArchitectures=native

# Resource limits
LimitNOFILE=65536
MemoryLimit=8G
CPUQuota=400%

# Environment
Environment=COMPOSE_FILE=/opt/container-analytics/deployment/docker/docker-compose.yml
Environment=COMPOSE_PROJECT_NAME=container-analytics

ExecStart=/usr/bin/docker-compose -f ${COMPOSE_FILE} up -d
ExecStop=/usr/bin/docker-compose -f ${COMPOSE_FILE} down
ExecReload=/usr/bin/docker-compose -f ${COMPOSE_FILE} restart

Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

    log "Hardened systemd service created at deployment/systemd/container-analytics-hardened.service"
}

# Set up monitoring for security events
setup_security_monitoring() {
    log "Setting up security monitoring..."
    
    # Create fail2ban filter for container analytics
    mkdir -p deployment/fail2ban
    
    cat > deployment/fail2ban/container-analytics.conf << 'EOF'
# Fail2Ban filter for Container Analytics
[Definition]
failregex = ^.*\[ERROR\].*Authentication failed.*<HOST>.*$
            ^.*\[ERROR\].*Access denied.*<HOST>.*$
            ^.*\[WARNING\].*Suspicious activity.*<HOST>.*$

ignoreregex =
EOF
    
    # Create fail2ban jail
    cat > deployment/fail2ban/container-analytics-jail.conf << 'EOF'
[container-analytics]
enabled = true
port = 8501,9090,3000
protocol = tcp
filter = container-analytics
logpath = /opt/container-analytics/logs/*.log
maxretry = 5
bantime = 3600
findtime = 600
EOF
    
    warn "Fail2ban configs created. Install with:"
    warn "sudo cp deployment/fail2ban/container-analytics.conf /etc/fail2ban/filter.d/"
    warn "sudo cp deployment/fail2ban/container-analytics-jail.conf /etc/fail2ban/jail.d/"
    
    log "Security monitoring configured"
}

# Create backup script with encryption
create_backup_script() {
    log "Creating encrypted backup script..."
    
    cat > deployment/backup-encrypted.sh << 'EOF'
#!/bin/bash
# Encrypted backup script for Container Analytics

set -euo pipefail

BACKUP_DIR="/opt/backups/container-analytics"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="container-analytics-backup-$DATE.tar.gz"
ENCRYPTED_FILE="$BACKUP_FILE.gpg"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Create backup
tar -czf "/tmp/$BACKUP_FILE" \
    --exclude='*.log' \
    --exclude='venv' \
    --exclude='.git' \
    data/ deployment/ *.py *.md requirements.txt .env

# Encrypt backup
gpg --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 \
    --s2k-digest-algo SHA512 --s2k-count 65536 \
    --symmetric --output "$BACKUP_DIR/$ENCRYPTED_FILE" \
    "/tmp/$BACKUP_FILE"

# Clean up temporary file
rm "/tmp/$BACKUP_FILE"

# Remove old backups (keep last 30 days)
find "$BACKUP_DIR" -name "*.gpg" -mtime +30 -delete

echo "Encrypted backup created: $BACKUP_DIR/$ENCRYPTED_FILE"
EOF
    
    chmod +x deployment/backup-encrypted.sh
    
    log "Encrypted backup script created at deployment/backup-encrypted.sh"
    warn "Set GPG passphrase and test the backup script before using in production"
}

# Audit Docker images for vulnerabilities
audit_docker_images() {
    log "Auditing Docker images for vulnerabilities..."
    
    if command -v docker >/dev/null 2>&1; then
        # Enable Docker Content Trust
        warn "Enable Docker Content Trust by setting DOCKER_CONTENT_TRUST=1"
        
        # Check for vulnerability scanning tools
        if command -v trivy >/dev/null 2>&1; then
            log "Scanning images with Trivy..."
            trivy image python:3.10-slim-bullseye
        else
            warn "Install Trivy for vulnerability scanning: https://aquasecurity.github.io/trivy/"
        fi
        
        # Check image signatures (if cosign is available)
        if command -v cosign >/dev/null 2>&1; then
            log "Cosign available for image signature verification"
        else
            warn "Consider installing cosign for image signature verification"
        fi
    fi
    
    log "Docker image audit completed"
}

# Main execution
main() {
    log "Starting Container Analytics Security Setup"
    
    check_permissions
    generate_passwords
    setup_ssl
    setup_file_permissions
    setup_docker_security
    setup_firewall
    setup_log_rotation
    create_hardened_systemd_service
    setup_security_monitoring
    create_backup_script
    audit_docker_images
    
    log "Security setup completed!"
    
    echo
    log "Next steps:"
    echo "1. Review and customize .env file with your specific settings"
    echo "2. Replace self-signed SSL certificate with CA-signed certificate"
    echo "3. Install fail2ban and logrotate configurations (see warnings above)"
    echo "4. Test the backup script and set up automated backups"
    echo "5. Review firewall rules and adjust as needed"
    echo "6. Enable Docker Content Trust: export DOCKER_CONTENT_TRUST=1"
    echo "7. Regularly update Docker images and scan for vulnerabilities"
}

# Run main function
main "$@"