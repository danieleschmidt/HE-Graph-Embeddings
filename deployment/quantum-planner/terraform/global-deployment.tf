# Quantum Task Planner - Global Multi-Region Deployment
# Production-grade infrastructure with auto-scaling, compliance, and quantum optimization

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket         = "quantum-planner-terraform-state"
    key            = "global/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "quantum-planner-terraform-locks"
  }
}

# Variables
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "regions" {
  description = "Regions for multi-region deployment"
  type        = list(string)
  default     = ["us-east-1", "eu-west-1", "ap-northeast-1", "us-west-2"]
}

variable "compliance_frameworks" {
  description = "Compliance frameworks to enable"
  type        = list(string)
  default     = ["GDPR", "CCPA", "HIPAA", "SOX", "PCI-DSS"]
}

variable "quantum_optimization_level" {
  description = "Quantum optimization level (1-5)"
  type        = number
  default     = 3
}

variable "enable_quantum_acceleration" {
  description = "Enable quantum acceleration features"
  type        = bool
  default     = true
}

variable "auto_scaling_enabled" {
  description = "Enable auto-scaling"
  type        = bool
  default     = true
}

variable "monitoring_level" {
  description = "Monitoring level (basic, standard, advanced)"
  type        = string
  default     = "advanced"
}

# Data sources
data "aws_availability_zones" "available" {
  for_each = toset(var.regions)
  
  provider = aws.region
  state    = "available"
}

data "aws_caller_identity" "current" {}

# Providers for multi-region deployment
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
  
  default_tags {
    tags = {
      Project           = "quantum-task-planner"
      Environment       = var.environment
      ManagedBy        = "terraform"
      QuantumOptimized = var.enable_quantum_acceleration
      ComplianceLevel  = join(",", var.compliance_frameworks)
    }
  }
}

provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
  
  default_tags {
    tags = {
      Project           = "quantum-task-planner"
      Environment       = var.environment
      ManagedBy        = "terraform"
      QuantumOptimized = var.enable_quantum_acceleration
      ComplianceLevel  = join(",", var.compliance_frameworks)
    }
  }
}

provider "aws" {
  alias  = "ap_northeast_1"
  region = "ap-northeast-1"
  
  default_tags {
    tags = {
      Project           = "quantum-task-planner"
      Environment       = var.environment
      ManagedBy        = "terraform"
      QuantumOptimized = var.enable_quantum_acceleration
      ComplianceLevel  = join(",", var.compliance_frameworks)
    }
  }
}

provider "aws" {
  alias  = "us_west_2"
  region = "us-west-2"
  
  default_tags {
    tags = {
      Project           = "quantum-task-planner"
      Environment       = var.environment
      ManagedBy        = "terraform"
      QuantumOptimized = var.enable_quantum_acceleration
      ComplianceLevel  = join(",", var.compliance_frameworks)
    }
  }
}

# Global Route53 hosted zone for DNS management
resource "aws_route53_zone" "quantum_planner_global" {
  provider = aws.us_east_1
  name     = "quantum-planner.ai"
  
  tags = {
    Name        = "quantum-planner-global-dns"
    Environment = var.environment
  }
}

# CloudFront distribution for global CDN
resource "aws_cloudfront_distribution" "quantum_planner_global" {
  provider = aws.us_east_1
  
  origin {
    domain_name = aws_lb.quantum_planner_alb["us-east-1"].dns_name
    origin_id   = "quantum-planner-primary"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  # Secondary origins for failover
  dynamic "origin" {
    for_each = [for region in var.regions : region if region != "us-east-1"]
    
    content {
      domain_name = aws_lb.quantum_planner_alb[origin.value].dns_name
      origin_id   = "quantum-planner-${origin.value}"
      
      custom_origin_config {
        http_port              = 80
        https_port             = 443
        origin_protocol_policy = "https-only"
        origin_ssl_protocols   = ["TLSv1.2"]
      }
    }
  }
  
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  
  # Global caching behavior
  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "quantum-planner-primary"
    compress              = true
    viewer_protocol_policy = "redirect-to-https"
    
    forwarded_values {
      query_string = true
      headers      = ["Authorization", "X-Quantum-Context", "X-Compliance-Region"]
      
      cookies {
        forward = "none"
      }
    }
    
    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }
  
  # API cache behavior
  ordered_cache_behavior {
    path_pattern     = "/api/quantum/*"
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD", "OPTIONS"]
    target_origin_id = "quantum-planner-primary"
    compress         = true
    
    viewer_protocol_policy = "https-only"
    
    forwarded_values {
      query_string = true
      headers      = ["*"]
      
      cookies {
        forward = "all"
      }
    }
    
    min_ttl     = 0
    default_ttl = 0  # No caching for API calls
    max_ttl     = 0
  }
  
  price_class = "PriceClass_All"
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.quantum_planner_cert.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
  
  tags = {
    Name        = "quantum-planner-global-cdn"
    Environment = var.environment
  }
}

# SSL Certificate for HTTPS
resource "aws_acm_certificate" "quantum_planner_cert" {
  provider    = aws.us_east_1
  domain_name = "quantum-planner.ai"
  
  subject_alternative_names = [
    "*.quantum-planner.ai",
    "api.quantum-planner.ai",
    "quantum.quantum-planner.ai"
  ]
  
  validation_method = "DNS"
  
  lifecycle {
    create_before_destroy = true
  }
  
  tags = {
    Name        = "quantum-planner-ssl-cert"
    Environment = var.environment
  }
}

# Regional VPCs
resource "aws_vpc" "quantum_planner_vpc" {
  for_each = toset(var.regions)
  
  provider             = aws.region
  cidr_block           = cidrsubnet("10.0.0.0/8", 8, index(var.regions, each.value))
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name                = "quantum-planner-vpc-${each.value}"
    Environment         = var.environment
    Region             = each.value
    "kubernetes.io/role/elb" = "1"
  }
}

# Public subnets for load balancers
resource "aws_subnet" "quantum_planner_public" {
  for_each = {
    for combo in setproduct(var.regions, [0, 1, 2]) :
    "${combo[0]}-${combo[1]}" => {
      region = combo[0]
      az_index = combo[1]
    }
  }
  
  provider                = aws.region
  vpc_id                  = aws_vpc.quantum_planner_vpc[each.value.region].id
  cidr_block              = cidrsubnet(aws_vpc.quantum_planner_vpc[each.value.region].cidr_block, 8, each.value.az_index)
  availability_zone       = data.aws_availability_zones.available[each.value.region].names[each.value.az_index]
  map_public_ip_on_launch = true
  
  tags = {
    Name                        = "quantum-planner-public-${each.value.region}-${each.value.az_index}"
    Environment                 = var.environment
    Type                       = "public"
    "kubernetes.io/role/elb"   = "1"
  }
}

# Private subnets for applications
resource "aws_subnet" "quantum_planner_private" {
  for_each = {
    for combo in setproduct(var.regions, [0, 1, 2]) :
    "${combo[0]}-${combo[1]}" => {
      region = combo[0]
      az_index = combo[1]
    }
  }
  
  provider          = aws.region
  vpc_id            = aws_vpc.quantum_planner_vpc[each.value.region].id
  cidr_block        = cidrsubnet(aws_vpc.quantum_planner_vpc[each.value.region].cidr_block, 8, each.value.az_index + 10)
  availability_zone = data.aws_availability_zones.available[each.value.region].names[each.value.az_index]
  
  tags = {
    Name                              = "quantum-planner-private-${each.value.region}-${each.value.az_index}"
    Environment                       = var.environment
    Type                             = "private"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# Internet Gateways
resource "aws_internet_gateway" "quantum_planner_igw" {
  for_each = toset(var.regions)
  
  provider = aws.region
  vpc_id   = aws_vpc.quantum_planner_vpc[each.value].id
  
  tags = {
    Name        = "quantum-planner-igw-${each.value}"
    Environment = var.environment
    Region      = each.value
  }
}

# EIP for NAT Gateways
resource "aws_eip" "quantum_planner_nat" {
  for_each = {
    for combo in setproduct(var.regions, [0, 1, 2]) :
    "${combo[0]}-${combo[1]}" => {
      region = combo[0]
      az_index = combo[1]
    }
  }
  
  provider = aws.region
  domain   = "vpc"
  
  tags = {
    Name        = "quantum-planner-nat-eip-${each.value.region}-${each.value.az_index}"
    Environment = var.environment
  }
  
  depends_on = [aws_internet_gateway.quantum_planner_igw]
}

# NAT Gateways
resource "aws_nat_gateway" "quantum_planner_nat" {
  for_each = {
    for combo in setproduct(var.regions, [0, 1, 2]) :
    "${combo[0]}-${combo[1]}" => {
      region = combo[0]
      az_index = combo[1]
    }
  }
  
  provider      = aws.region
  allocation_id = aws_eip.quantum_planner_nat[each.key].id
  subnet_id     = aws_subnet.quantum_planner_public[each.key].id
  
  tags = {
    Name        = "quantum-planner-nat-${each.value.region}-${each.value.az_index}"
    Environment = var.environment
  }
  
  depends_on = [aws_internet_gateway.quantum_planner_igw]
}

# Route tables
resource "aws_route_table" "quantum_planner_public" {
  for_each = toset(var.regions)
  
  provider = aws.region
  vpc_id   = aws_vpc.quantum_planner_vpc[each.value].id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.quantum_planner_igw[each.value].id
  }
  
  tags = {
    Name        = "quantum-planner-public-rt-${each.value}"
    Environment = var.environment
    Type        = "public"
  }
}

resource "aws_route_table" "quantum_planner_private" {
  for_each = {
    for combo in setproduct(var.regions, [0, 1, 2]) :
    "${combo[0]}-${combo[1]}" => {
      region = combo[0]
      az_index = combo[1]
    }
  }
  
  provider = aws.region
  vpc_id   = aws_vpc.quantum_planner_vpc[each.value.region].id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.quantum_planner_nat[each.key].id
  }
  
  tags = {
    Name        = "quantum-planner-private-rt-${each.value.region}-${each.value.az_index}"
    Environment = var.environment
    Type        = "private"
  }
}

# Route table associations
resource "aws_route_table_association" "quantum_planner_public" {
  for_each = {
    for combo in setproduct(var.regions, [0, 1, 2]) :
    "${combo[0]}-${combo[1]}" => {
      region = combo[0]
      az_index = combo[1]
    }
  }
  
  provider       = aws.region
  subnet_id      = aws_subnet.quantum_planner_public[each.key].id
  route_table_id = aws_route_table.quantum_planner_public[each.value.region].id
}

resource "aws_route_table_association" "quantum_planner_private" {
  for_each = {
    for combo in setproduct(var.regions, [0, 1, 2]) :
    "${combo[0]}-${combo[1]}" => {
      region = combo[0]
      az_index = combo[1]
    }
  }
  
  provider       = aws.region
  subnet_id      = aws_subnet.quantum_planner_private[each.key].id
  route_table_id = aws_route_table.quantum_planner_private[each.key].id
}

# Security groups
resource "aws_security_group" "quantum_planner_alb" {
  for_each = toset(var.regions)
  
  provider    = aws.region
  name        = "quantum-planner-alb-${each.value}"
  description = "Security group for Quantum Task Planner ALB"
  vpc_id      = aws_vpc.quantum_planner_vpc[each.value].id
  
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "quantum-planner-alb-sg-${each.value}"
    Environment = var.environment
  }
}

resource "aws_security_group" "quantum_planner_eks" {
  for_each = toset(var.regions)
  
  provider    = aws.region
  name        = "quantum-planner-eks-${each.value}"
  description = "Security group for Quantum Task Planner EKS cluster"
  vpc_id      = aws_vpc.quantum_planner_vpc[each.value].id
  
  ingress {
    description = "HTTPS API server"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.quantum_planner_vpc[each.value].cidr_block]
  }
  
  # Quantum communication ports
  ingress {
    description = "Quantum entanglement sync"
    from_port   = 9090
    to_port     = 9099
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.quantum_planner_vpc[each.value].cidr_block]
  }
  
  ingress {
    description = "Quantum state monitoring"
    from_port   = 9100
    to_port     = 9109
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.quantum_planner_vpc[each.value].cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "quantum-planner-eks-sg-${each.value}"
    Environment = var.environment
  }
}

# Application Load Balancers
resource "aws_lb" "quantum_planner_alb" {
  for_each = toset(var.regions)
  
  provider           = aws.region
  name               = "quantum-planner-alb-${each.value}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.quantum_planner_alb[each.value].id]
  
  subnets = [
    for idx in [0, 1, 2] : aws_subnet.quantum_planner_public["${each.value}-${idx}"].id
  ]
  
  enable_deletion_protection       = true
  enable_cross_zone_load_balancing = true
  enable_http2                     = true
  
  access_logs {
    bucket  = aws_s3_bucket.quantum_planner_logs[each.value].bucket
    prefix  = "alb-logs"
    enabled = true
  }
  
  tags = {
    Name        = "quantum-planner-alb-${each.value}"
    Environment = var.environment
    Region      = each.value
  }
}

# S3 buckets for logs and data
resource "aws_s3_bucket" "quantum_planner_logs" {
  for_each = toset(var.regions)
  
  provider = aws.region
  bucket   = "quantum-planner-logs-${each.value}-${random_string.bucket_suffix.result}"
  
  tags = {
    Name        = "quantum-planner-logs-${each.value}"
    Environment = var.environment
    Purpose     = "logging"
  }
}

resource "aws_s3_bucket_versioning" "quantum_planner_logs_versioning" {
  for_each = toset(var.regions)
  
  provider = aws.region
  bucket   = aws_s3_bucket.quantum_planner_logs[each.value].id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "quantum_planner_logs_encryption" {
  for_each = toset(var.regions)
  
  provider = aws.region
  bucket   = aws_s3_bucket.quantum_planner_logs[each.value].id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Random string for unique bucket names
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# EKS Clusters
resource "aws_eks_cluster" "quantum_planner" {
  for_each = toset(var.regions)
  
  provider = aws.region
  name     = "quantum-planner-${each.value}"
  role_arn = aws_iam_role.quantum_planner_eks_cluster[each.value].arn
  version  = "1.28"
  
  vpc_config {
    subnet_ids = concat(
      [for idx in [0, 1, 2] : aws_subnet.quantum_planner_public["${each.value}-${idx}"].id],
      [for idx in [0, 1, 2] : aws_subnet.quantum_planner_private["${each.value}-${idx}"].id]
    )
    
    security_group_ids      = [aws_security_group.quantum_planner_eks[each.value].id]
    endpoint_private_access = true
    endpoint_public_access  = true
    
    public_access_cidrs = ["0.0.0.0/0"]
  }
  
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  encryption_config {
    provider {
      key_arn = aws_kms_key.quantum_planner_eks[each.value].arn
    }
    resources = ["secrets"]
  }
  
  tags = {
    Name        = "quantum-planner-eks-${each.value}"
    Environment = var.environment
    Region      = each.value
    QuantumOptimized = var.enable_quantum_acceleration
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.quantum_planner_eks_cluster_policy,
    aws_iam_role_policy_attachment.quantum_planner_eks_service_policy,
    aws_cloudwatch_log_group.quantum_planner_eks
  ]
}

# EKS Node Groups
resource "aws_eks_node_group" "quantum_planner" {
  for_each = toset(var.regions)
  
  provider        = aws.region
  cluster_name    = aws_eks_cluster.quantum_planner[each.value].name
  node_group_name = "quantum-planner-nodes"
  node_role_arn   = aws_iam_role.quantum_planner_eks_node_group[each.value].arn
  
  subnet_ids = [
    for idx in [0, 1, 2] : aws_subnet.quantum_planner_private["${each.value}-${idx}"].id
  ]
  
  capacity_type  = "ON_DEMAND"
  instance_types = ["m5.xlarge", "m5.2xlarge", "c5.2xlarge"]  # Quantum-optimized instances
  
  scaling_config {
    desired_size = var.auto_scaling_enabled ? 3 : 2
    max_size     = var.auto_scaling_enabled ? 10 : 5
    min_size     = 2
  }
  
  update_config {
    max_unavailable = 1
  }
  
  # Quantum-optimized node configuration
  user_data = base64encode(templatefile("${path.module}/templates/node-userdata.sh", {
    quantum_optimization_level = var.quantum_optimization_level
    enable_gpu_support        = true
    cluster_endpoint          = aws_eks_cluster.quantum_planner[each.value].endpoint
    cluster_ca                = aws_eks_cluster.quantum_planner[each.value].certificate_authority[0].data
    cluster_name              = aws_eks_cluster.quantum_planner[each.value].name
  }))
  
  tags = {
    Name                = "quantum-planner-node-group-${each.value}"
    Environment         = var.environment
    QuantumOptimized   = var.enable_quantum_acceleration
    "kubernetes.io/cluster/quantum-planner-${each.value}" = "owned"
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.quantum_planner_eks_worker_node_policy,
    aws_iam_role_policy_attachment.quantum_planner_eks_cni_policy,
    aws_iam_role_policy_attachment.quantum_planner_eks_container_registry_policy,
  ]
}

# KMS keys for encryption
resource "aws_kms_key" "quantum_planner_eks" {
  for_each = toset(var.regions)
  
  provider    = aws.region
  description = "KMS key for Quantum Task Planner EKS cluster encryption"
  
  tags = {
    Name        = "quantum-planner-eks-kms-${each.value}"
    Environment = var.environment
  }
}

resource "aws_kms_alias" "quantum_planner_eks" {
  for_each = toset(var.regions)
  
  provider      = aws.region
  name          = "alias/quantum-planner-eks-${each.value}"
  target_key_id = aws_kms_key.quantum_planner_eks[each.value].key_id
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "quantum_planner_eks" {
  for_each = toset(var.regions)
  
  provider          = aws.region
  name              = "/aws/eks/quantum-planner-${each.value}/cluster"
  retention_in_days = 30
  
  tags = {
    Name        = "quantum-planner-eks-logs-${each.value}"
    Environment = var.environment
  }
}

# IAM roles and policies
resource "aws_iam_role" "quantum_planner_eks_cluster" {
  for_each = toset(var.regions)
  
  provider = aws.region
  name     = "quantum-planner-eks-cluster-role-${each.value}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "quantum_planner_eks_cluster_policy" {
  for_each = toset(var.regions)
  
  provider   = aws.region
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.quantum_planner_eks_cluster[each.value].name
}

resource "aws_iam_role_policy_attachment" "quantum_planner_eks_service_policy" {
  for_each = toset(var.regions)
  
  provider   = aws.region
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSServicePolicy"
  role       = aws_iam_role.quantum_planner_eks_cluster[each.value].name
}

resource "aws_iam_role" "quantum_planner_eks_node_group" {
  for_each = toset(var.regions)
  
  provider = aws.region
  name     = "quantum-planner-eks-node-group-role-${each.value}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "quantum_planner_eks_worker_node_policy" {
  for_each = toset(var.regions)
  
  provider   = aws.region
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.quantum_planner_eks_node_group[each.value].name
}

resource "aws_iam_role_policy_attachment" "quantum_planner_eks_cni_policy" {
  for_each = toset(var.regions)
  
  provider   = aws.region
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.quantum_planner_eks_node_group[each.value].name
}

resource "aws_iam_role_policy_attachment" "quantum_planner_eks_container_registry_policy" {
  for_each = toset(var.regions)
  
  provider   = aws.region
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.quantum_planner_eks_node_group[each.value].name
}

# Outputs
output "eks_cluster_endpoints" {
  description = "EKS cluster endpoints by region"
  value = {
    for region in var.regions :
    region => aws_eks_cluster.quantum_planner[region].endpoint
  }
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = aws_cloudfront_distribution.quantum_planner_global.id
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  value       = aws_cloudfront_distribution.quantum_planner_global.domain_name
}

output "route53_zone_id" {
  description = "Route53 hosted zone ID"
  value       = aws_route53_zone.quantum_planner_global.zone_id
}

output "load_balancer_dns_names" {
  description = "Application Load Balancer DNS names by region"
  value = {
    for region in var.regions :
    region => aws_lb.quantum_planner_alb[region].dns_name
  }
}

output "s3_log_buckets" {
  description = "S3 log bucket names by region"
  value = {
    for region in var.regions :
    region => aws_s3_bucket.quantum_planner_logs[region].bucket
  }
}

output "quantum_planner_vpc_ids" {
  description = "VPC IDs by region"
  value = {
    for region in var.regions :
    region => aws_vpc.quantum_planner_vpc[region].id
  }
}