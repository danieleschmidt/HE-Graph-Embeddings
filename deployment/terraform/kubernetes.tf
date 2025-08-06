# Kubernetes deployment configuration for HE-Graph-Embeddings
# EKS clusters in multiple regions with auto-scaling and compliance

# EKS Clusters
resource "aws_eks_cluster" "main" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  name     = "${var.project_name}-${each.value}"
  role_arn = aws_iam_role.eks_cluster[each.value].arn
  version  = "1.27"
  
  vpc_config {
    subnet_ids = [
      aws_subnet.private["${each.value}-a"].id,
      aws_subnet.private["${each.value}-b"].id,
      aws_subnet.private["${each.value}-c"].id,
    ]
    
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs    = ["0.0.0.0/0"]
  }
  
  encryption_config {
    provider {
      key_arn = aws_kms_key.he_graph_key[each.value].arn
    }
    resources = ["secrets"]
  }
  
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-eks-${each.value}"
    Region = each.value
  })
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
  ]
}

# EKS Cluster IAM Role
resource "aws_iam_role" "eks_cluster" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  name = "${var.project_name}-eks-cluster-${each.value}"
  
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
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-eks-cluster-role-${each.value}"
    Region = each.value
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster[each.value].name
}

resource "aws_iam_role_policy_attachment" "eks_vpc_resource_controller" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
  role       = aws_iam_role.eks_cluster[each.value].name
}

# EKS Node Groups
resource "aws_eks_node_group" "main" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  cluster_name    = aws_eks_cluster.main[each.value].name
  node_group_name = "${var.project_name}-nodes"
  node_role_arn   = aws_iam_role.eks_node_group[each.value].arn
  
  subnet_ids = [
    aws_subnet.private["${each.value}-a"].id,
    aws_subnet.private["${each.value}-b"].id,
    aws_subnet.private["${each.value}-c"].id,
  ]
  
  capacity_type  = "ON_DEMAND"
  instance_types = var.enable_gpu_instances ? ["g4dn.xlarge", "g4dn.2xlarge"] : ["m5.large", "m5.xlarge"]
  
  scaling_config {
    desired_size = var.min_replicas
    max_size     = var.max_replicas
    min_size     = var.min_replicas
  }
  
  update_config {
    max_unavailable = 1
  }
  
  # Ensure proper ordering of resource creation
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.ec2_container_registry_read_only,
  ]
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-nodes-${each.value}"
    Region = each.value
  })
}

# GPU Node Group (if enabled)
resource "aws_eks_node_group" "gpu" {
  for_each = var.enable_gpu_instances ? toset(var.regions) : []
  
  provider = aws.region[each.value]
  
  cluster_name    = aws_eks_cluster.main[each.value].name
  node_group_name = "${var.project_name}-gpu-nodes"
  node_role_arn   = aws_iam_role.eks_node_group[each.value].arn
  
  subnet_ids = [
    aws_subnet.private["${each.value}-a"].id,
    aws_subnet.private["${each.value}-b"].id,
    aws_subnet.private["${each.value}-c"].id,
  ]
  
  capacity_type  = "ON_DEMAND"
  instance_types = ["p3.2xlarge", "p3.8xlarge"]
  
  scaling_config {
    desired_size = 0  # Start with 0, scale up on demand
    max_size     = 3
    min_size     = 0
  }
  
  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.ec2_container_registry_read_only,
  ]
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-gpu-nodes-${each.value}"
    Region = each.value
    Type   = "gpu"
  })
}

# EKS Node Group IAM Role
resource "aws_iam_role" "eks_node_group" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  name = "${var.project_name}-eks-node-group-${each.value}"
  
  assume_role_policy = jsonencode({
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
    Version = "2012-10-17"
  })
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-eks-node-group-role-${each.value}"
    Region = each.value
  })
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node_group[each.value].name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node_group[each.value].name
}

resource "aws_iam_role_policy_attachment" "ec2_container_registry_read_only" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node_group[each.value].name
}

# Additional IAM policy for HE-Graph-Embeddings specific permissions
resource "aws_iam_policy" "he_graph_permissions" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  name        = "${var.project_name}-permissions-${each.value}"
  description = "Additional permissions for HE-Graph-Embeddings"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.he_graph_key[each.value].arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.logs[each.value].arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-permissions-${each.value}"
    Region = each.value
  })
}

resource "aws_iam_role_policy_attachment" "he_graph_permissions" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  policy_arn = aws_iam_policy.he_graph_permissions[each.value].arn
  role       = aws_iam_role.eks_node_group[each.value].name
}

# EKS OIDC Identity Provider
data "tls_certificate" "eks_oidc" {
  for_each = toset(var.regions)
  
  provider = tls.region[each.value]
  
  url = aws_eks_cluster.main[each.value].identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks_oidc" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks_oidc[each.value].certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.main[each.value].identity[0].oidc[0].issuer
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-eks-oidc-${each.value}"
    Region = each.value
  })
}

# EKS Addons
resource "aws_eks_addon" "vpc_cni" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  cluster_name      = aws_eks_cluster.main[each.value].name
  addon_name        = "vpc-cni"
  addon_version     = "v1.13.4-eksbuild.1"
  resolve_conflicts = "OVERWRITE"
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-vpc-cni-${each.value}"
    Region = each.value
  })
}

resource "aws_eks_addon" "coredns" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  cluster_name      = aws_eks_cluster.main[each.value].name
  addon_name        = "coredns"
  addon_version     = "v1.10.1-eksbuild.1"
  resolve_conflicts = "OVERWRITE"
  
  depends_on = [aws_eks_node_group.main]
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-coredns-${each.value}"
    Region = each.value
  })
}

resource "aws_eks_addon" "kube_proxy" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  cluster_name      = aws_eks_cluster.main[each.value].name
  addon_name        = "kube-proxy"
  addon_version     = "v1.27.3-eksbuild.1"
  resolve_conflicts = "OVERWRITE"
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-kube-proxy-${each.value}"
    Region = each.value
  })
}

# EBS CSI Driver addon (for persistent volumes)
resource "aws_eks_addon" "ebs_csi_driver" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  cluster_name             = aws_eks_cluster.main[each.value].name
  addon_name               = "aws-ebs-csi-driver"
  addon_version            = "v1.20.0-eksbuild.1"
  service_account_role_arn = aws_iam_role.ebs_csi_driver[each.value].arn
  resolve_conflicts        = "OVERWRITE"
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-ebs-csi-${each.value}"
    Region = each.value
  })
}

# IAM role for EBS CSI driver
resource "aws_iam_role" "ebs_csi_driver" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  name = "${var.project_name}-ebs-csi-driver-${each.value}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.eks_oidc[each.value].arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(aws_iam_openid_connect_provider.eks_oidc[each.value].url, "https://", "")}:sub" = "system:serviceaccount:kube-system:ebs-csi-controller-sa"
            "${replace(aws_iam_openid_connect_provider.eks_oidc[each.value].url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-ebs-csi-driver-role-${each.value}"
    Region = each.value
  })
}

resource "aws_iam_role_policy_attachment" "ebs_csi_driver" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  policy_arn = "arn:aws:iam::aws:policy/service-role/Amazon_EBS_CSI_DriverPolicy"
  role       = aws_iam_role.ebs_csi_driver[each.value].name
}

# CloudWatch log groups for EKS
resource "aws_cloudwatch_log_group" "eks" {
  for_each = toset(var.regions)
  
  provider = aws.region[each.value]
  
  name              = "/aws/eks/${var.project_name}-${each.value}/cluster"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.he_graph_key[each.value].arn
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-eks-logs-${each.value}"
    Region = each.value
  })
}

# Output EKS cluster information
output "eks_cluster_endpoints" {
  description = "EKS cluster endpoints"
  value = {
    for region in var.regions :
    region => aws_eks_cluster.main[region].endpoint
  }
}

output "eks_cluster_security_group_ids" {
  description = "EKS cluster security group IDs"
  value = {
    for region in var.regions :
    region => aws_eks_cluster.main[region].vpc_config[0].cluster_security_group_id
  }
}

output "eks_cluster_oidc_issuer_urls" {
  description = "EKS cluster OIDC issuer URLs"
  value = {
    for region in var.regions :
    region => aws_eks_cluster.main[region].identity[0].oidc[0].issuer
  }
}