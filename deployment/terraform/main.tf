# HE-Graph-Embeddings Multi-Region Production Deployment
# Terraform configuration for global deployment with compliance and auto-scaling

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
  
  backend "s3" {
    bucket = var.terraform_state_bucket
    key    = "he-graph-embeddings/terraform.tfstate"
    region = var.primary_region
    encrypt = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Variables
variable "project_name" {
  description = "Project name"
  type        = string
  default     = "he-graph-embeddings"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "primary_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-east-1"
}

variable "regions" {
  description = "List of regions for multi-region deployment"
  type        = list(string)
  default     = ["us-east-1", "eu-west-1", "ap-northeast-1"]
}

variable "compliance_frameworks" {
  description = "Compliance frameworks to enable"
  type        = list(string)
  default     = ["GDPR", "CCPA", "HIPAA", "SOC2"]
}

variable "enable_gpu_instances" {
  description = "Enable GPU instances for HE computation"
  type        = bool
  default     = true
}

variable "min_replicas" {
  description = "Minimum number of replicas per region"
  type        = number
  default     = 2
}

variable "max_replicas" {
  description = "Maximum number of replicas per region"
  type        = number
  default     = 10
}

variable "terraform_state_bucket" {
  description = "S3 bucket for Terraform state"
  type        = string
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Component   = "HE-Graph-Embeddings"
  }
  
  # Compliance configurations
  compliance_config = {
    GDPR = {
      data_retention_days = 90
      encryption_required = true
      audit_logging      = true
      data_residency     = ["eu-west-1", "eu-central-1"]
    }
    CCPA = {
      data_retention_days = 365
      encryption_required = true
      audit_logging      = true
      data_residency     = ["us-west-1", "us-east-1"]
    }
    HIPAA = {
      data_retention_days = 2555  # 7 years
      encryption_required = true
      audit_logging      = true
      dedicated_tenancy  = true
    }
    SOC2 = {
      audit_logging      = true
      access_logging     = true
      change_management  = true
      encryption_required = true
    }
  }
}

# KMS keys for encryption
resource "aws_kms_key" "he_graph_key" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  description             = "HE-Graph-Embeddings encryption key - ${each.value}"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow use of the key for HE-Graph services"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.project_name}-*"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-${each.value}"
    Region = each.value
  })
}

resource "aws_kms_alias" "he_graph_key_alias" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  name          = "alias/${var.project_name}-${each.value}"
  target_key_id = aws_kms_key.he_graph_key[each.value].key_id
}

# VPC and networking for each region
resource "aws_vpc" "main" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  cidr_block           = "10.${index(var.regions, each.value)}.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-${each.value}"
    Region = each.value
  })
}

resource "aws_internet_gateway" "main" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  vpc_id = aws_vpc.main[each.value].id
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-igw-${each.value}"
    Region = each.value
  })
}

# Public subnets for load balancers
resource "aws_subnet" "public" {
  for_each = {
    for pair in setproduct(var.regions, ["a", "b", "c"]) :
    "${pair[0]}-${pair[1]}" => {
      region = pair[0]
      az     = pair[1]
    }
  }
  
  provider = aws.region[each.value.region]
  
  vpc_id                  = aws_vpc.main[each.value.region].id
  cidr_block              = "10.${index(var.regions, each.value.region)}.${each.value.az == "a" ? 1 : each.value.az == "b" ? 2 : 3}.0/24"
  availability_zone       = "${each.value.region}${each.value.az}"
  map_public_ip_on_launch = true
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-public-${each.value.region}${each.value.az}"
    Region = each.value.region
    Type   = "public"
  })
}

# Private subnets for application instances
resource "aws_subnet" "private" {
  for_each = {
    for pair in setproduct(var.regions, ["a", "b", "c"]) :
    "${pair[0]}-${pair[1]}" => {
      region = pair[0]
      az     = pair[1]
    }
  }
  
  provider = aws.region[each.value.region]
  
  vpc_id            = aws_vpc.main[each.value.region].id
  cidr_block        = "10.${index(var.regions, each.value.region)}.${each.value.az == "a" ? 11 : each.value.az == "b" ? 12 : 13}.0/24"
  availability_zone = "${each.value.region}${each.value.az}"
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-private-${each.value.region}${each.value.az}"
    Region = each.value.region
    Type   = "private"
  })
}

# NAT Gateways for private subnet internet access
resource "aws_eip" "nat" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  domain = "vpc"
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-nat-${each.value}"
    Region = each.value
  })
  
  depends_on = [aws_internet_gateway.main]
}

resource "aws_nat_gateway" "main" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  allocation_id = aws_eip.nat[each.value].id
  subnet_id     = aws_subnet.public["${each.value}-a"].id
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-nat-${each.value}"
    Region = each.value
  })
  
  depends_on = [aws_internet_gateway.main]
}

# Route tables
resource "aws_route_table" "public" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  vpc_id = aws_vpc.main[each.value].id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main[each.value].id
  }
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-public-${each.value}"
    Region = each.value
  })
}

resource "aws_route_table" "private" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  vpc_id = aws_vpc.main[each.value].id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[each.value].id
  }
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-private-${each.value}"
    Region = each.value
  })
}

# Route table associations
resource "aws_route_table_association" "public" {
  for_each = {
    for pair in setproduct(var.regions, ["a", "b", "c"]) :
    "${pair[0]}-${pair[1]}" => {
      region = pair[0]
      az     = pair[1]
    }
  }
  
  provider = aws.region[each.value.region]
  
  subnet_id      = aws_subnet.public[each.key].id
  route_table_id = aws_route_table.public[each.value.region].id
}

resource "aws_route_table_association" "private" {
  for_each = {
    for pair in setproduct(var.regions, ["a", "b", "c"]) :
    "${pair[0]}-${pair[1]}" => {
      region = pair[0]
      az     = pair[1]
    }
  }
  
  provider = aws.region[each.value.region]
  
  subnet_id      = aws_subnet.private[each.key].id
  route_table_id = aws_route_table.private[each.value.region].id
}

# Security groups
resource "aws_security_group" "alb" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  name_prefix = "${var.project_name}-alb-"
  vpc_id      = aws_vpc.main[each.value].id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
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
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-alb-${each.value}"
    Region = each.value
  })
}

resource "aws_security_group" "app" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  name_prefix = "${var.project_name}-app-"
  vpc_id      = aws_vpc.main[each.value].id
  
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb[each.value].id]
  }
  
  ingress {
    from_port       = 9090  # Metrics port
    to_port         = 9090
    protocol        = "tcp"
    security_groups = [aws_security_group.alb[each.value].id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-app-${each.value}"
    Region = each.value
  })
}

# Application Load Balancers
resource "aws_lb" "main" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  name               = "${var.project_name}-${each.value}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb[each.value].id]
  
  subnets = [
    aws_subnet.public["${each.value}-a"].id,
    aws_subnet.public["${each.value}-b"].id,
    aws_subnet.public["${each.value}-c"].id,
  ]
  
  enable_deletion_protection = var.environment == "prod"
  
  access_logs {
    bucket  = aws_s3_bucket.logs[each.value].id
    prefix  = "alb"
    enabled = true
  }
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-alb-${each.value}"
    Region = each.value
  })
  
  depends_on = [aws_s3_bucket_policy.logs]
}

# S3 buckets for logging
resource "aws_s3_bucket" "logs" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  bucket        = "${var.project_name}-logs-${each.value}-${random_string.bucket_suffix.result}"
  force_destroy = var.environment != "prod"
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-logs-${each.value}"
    Region = each.value
    Purpose = "logging"
  })
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_versioning" "logs" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  bucket = aws_s3_bucket.logs[each.value].id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "logs" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  bucket = aws_s3_bucket.logs[each.value].id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.he_graph_key[each.value].arn
        sse_algorithm     = "aws:kms"
      }
    }
  }
}

resource "aws_s3_bucket_policy" "logs" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  bucket = aws_s3_bucket.logs[each.value].id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AWSLogDeliveryWrite"
        Effect    = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_elb_service_account.main[each.value].id}:root"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.logs[each.value].arn}/alb/AWSLogs/${data.aws_caller_identity.current.account_id}/*"
      },
      {
        Sid       = "AWSLogDeliveryCheck"
        Effect    = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_elb_service_account.main[each.value].id}:root"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.logs[each.value].arn
      }
    ]
  })
}

data "aws_elb_service_account" "main" {
  for_each = toset(var.regions)
  provider = aws.region[each.value]
}

# Output values
output "vpc_ids" {
  description = "VPC IDs by region"
  value = {
    for region in var.regions :
    region => aws_vpc.main[region].id
  }
}

output "load_balancer_dns_names" {
  description = "Load balancer DNS names by region"
  value = {
    for region in var.regions :
    region => aws_lb.main[region].dns_name
  }
}

output "kms_key_ids" {
  description = "KMS key IDs by region"
  value = {
    for region in var.regions :
    region => aws_kms_key.he_graph_key[region].key_id
  }
}

output "s3_log_buckets" {
  description = "S3 log bucket names by region"
  value = {
    for region in var.regions :
    region => aws_s3_bucket.logs[region].id
  }
}

# Provider configurations for multiple regions
provider "aws" {
  alias  = "us-east-1"
  region = "us-east-1"
  
  default_tags {
    tags = local.common_tags
  }
}

provider "aws" {
  alias  = "eu-west-1"
  region = "eu-west-1"
  
  default_tags {
    tags = local.common_tags
  }
}

provider "aws" {
  alias  = "ap-northeast-1"
  region = "ap-northeast-1"
  
  default_tags {
    tags = local.common_tags
  }
}

# Dynamic provider configuration for each region
provider "aws" {
  for_each = toset(var.regions)
  alias    = "region"
  region   = each.value
  
  default_tags {
    tags = merge(local.common_tags, {
      Region = each.value
    })
  }
}