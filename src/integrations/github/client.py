"""
GitHub API client for HE-Graph-Embeddings
"""

import os
import logging
from typing import Dict, List, Optional, Any
import httpx
import asyncio
from datetime import datetime, timezone
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GitHubConfig:
    """GitHub API configuration"""
    token: str
    base_url: str = "https://api.github.com"
    timeout: int = 30
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> 'GitHubConfig':
        """Create config from environment variables"""
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
        
        return cls(
            token=token,
            base_url=os.getenv('GITHUB_API_URL', "https://api.github.com"),
            timeout=int(os.getenv('GITHUB_TIMEOUT', '30')),
            max_retries=int(os.getenv('GITHUB_MAX_RETRIES', '3'))
        )

class GitHubClient:
    """GitHub API client"""
    
    def __init__(self, config: GitHubConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "HE-Graph-Embeddings/1.0"
        }
        
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to GitHub API"""
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            for attempt in range(self.config.max_retries):
                try:
                    response = await client.request(
                        method, url, headers=self.headers, **kwargs
                    )
                    
                    if response.status_code == 401:
                        raise GitHubAuthError("Invalid or expired GitHub token")
                    
                    if response.status_code == 403:
                        rate_limit_reset = response.headers.get('x-ratelimit-reset')
                        if rate_limit_reset:
                            reset_time = datetime.fromtimestamp(int(rate_limit_reset), timezone.utc)
                            raise GitHubRateLimitError(f"Rate limit exceeded. Resets at {reset_time}")
                        raise GitHubAuthError("Access forbidden")
                    
                    if response.status_code == 404:
                        raise GitHubNotFoundError(f"Resource not found: {endpoint}")
                    
                    response.raise_for_status()
                    
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        return {"content": response.text}
                        
                except httpx.RequestError as e:
                    if attempt == self.config.max_retries - 1:
                        raise GitHubClientError(f"Request failed after {self.config.max_retries} attempts: {e}")
                    
                    wait_time = 2 ** attempt
                    logger.warning(f"GitHub API request failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
        
        raise GitHubClientError("Max retries exceeded")
    
    async def get_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository information"""
        return await self._request("GET", f"/repos/{owner}/{repo}")
    
    async def list_repositories(self, owner: str, repo_type: str = "all") -> List[Dict[str, Any]]:
        """List repositories for a user or organization"""
        endpoint = f"/users/{owner}/repos" if repo_type == "user" else f"/orgs/{owner}/repos"
        return await self._request("GET", endpoint, params={"type": repo_type})
    
    async def get_commits(self, owner: str, repo: str, 
                         since: Optional[datetime] = None,
                         until: Optional[datetime] = None,
                         per_page: int = 30) -> List[Dict[str, Any]]:
        """Get repository commits"""
        params = {"per_page": per_page}
        if since:
            params["since"] = since.isoformat()
        if until:
            params["until"] = until.isoformat()
            
        return await self._request("GET", f"/repos/{owner}/{repo}/commits", params=params)
    
    async def get_issues(self, owner: str, repo: str, 
                        state: str = "open",
                        labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get repository issues"""
        params = {"state": state}
        if labels:
            params["labels"] = ",".join(labels)
            
        return await self._request("GET", f"/repos/{owner}/{repo}/issues", params=params)
    
    async def create_issue(self, owner: str, repo: str, 
                          title: str, body: str,
                          labels: Optional[List[str]] = None,
                          assignees: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new issue"""
        data = {"title": title, "body": body}
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees
            
        return await self._request("POST", f"/repos/{owner}/{repo}/issues", json=data)
    
    async def get_pull_requests(self, owner: str, repo: str,
                               state: str = "open") -> List[Dict[str, Any]]:
        """Get repository pull requests"""
        return await self._request("GET", f"/repos/{owner}/{repo}/pulls", 
                                 params={"state": state})
    
    async def create_pull_request(self, owner: str, repo: str,
                                 title: str, body: str,
                                 head: str, base: str = "main") -> Dict[str, Any]:
        """Create a new pull request"""
        data = {
            "title": title,
            "body": body,
            "head": head,
            "base": base
        }
        
        return await self._request("POST", f"/repos/{owner}/{repo}/pulls", json=data)
    
    async def get_releases(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """Get repository releases"""
        return await self._request("GET", f"/repos/{owner}/{repo}/releases")
    
    async def create_release(self, owner: str, repo: str,
                           tag_name: str, name: str,
                           body: str, prerelease: bool = False) -> Dict[str, Any]:
        """Create a new release"""
        data = {
            "tag_name": tag_name,
            "name": name,
            "body": body,
            "prerelease": prerelease
        }
        
        return await self._request("POST", f"/repos/{owner}/{repo}/releases", json=data)
    
    async def get_workflow_runs(self, owner: str, repo: str,
                               workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get GitHub Actions workflow runs"""
        endpoint = f"/repos/{owner}/{repo}/actions/runs"
        if workflow_id:
            endpoint = f"/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs"
            
        return await self._request("GET", endpoint)
    
    async def trigger_workflow(self, owner: str, repo: str, 
                              workflow_id: str, ref: str = "main",
                              inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Trigger a workflow dispatch"""
        data = {"ref": ref}
        if inputs:
            data["inputs"] = inputs
            
        return await self._request("POST", 
                                 f"/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches",
                                 json=data)
    
    async def get_repository_contents(self, owner: str, repo: str, 
                                    path: str = "", ref: str = "main") -> Dict[str, Any]:
        """Get repository file contents"""
        return await self._request("GET", f"/repos/{owner}/{repo}/contents/{path}",
                                 params={"ref": ref})
    
    async def create_or_update_file(self, owner: str, repo: str, path: str,
                                   message: str, content: str,
                                   sha: Optional[str] = None,
                                   branch: str = "main") -> Dict[str, Any]:
        """Create or update a file in repository"""
        import base64
        
        data = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch
        }
        
        if sha:  # Update existing file
            data["sha"] = sha
            
        return await self._request("PUT", f"/repos/{owner}/{repo}/contents/{path}", json=data)
    
    async def get_rate_limit(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return await self._request("GET", "/rate_limit")
    
    async def get_user(self) -> Dict[str, Any]:
        """Get authenticated user information"""
        return await self._request("GET", "/user")
    
    async def search_repositories(self, query: str, 
                                sort: str = "stars",
                                order: str = "desc") -> Dict[str, Any]:
        """Search repositories"""
        return await self._request("GET", "/search/repositories",
                                 params={"q": query, "sort": sort, "order": order})
    
    async def get_repository_statistics(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository statistics"""
        # Get various stats in parallel
        tasks = [
            self.get_repository(owner, repo),
            self.get_commits(owner, repo, per_page=1),
            self.get_issues(owner, repo, state="all"),
            self.get_pull_requests(owner, repo, state="all"),
            self.get_releases(owner, repo)
        ]
        
        repo_info, commits, issues, prs, releases = await asyncio.gather(*tasks)
        
        return {
            "repository": repo_info,
            "total_commits": len(commits),
            "total_issues": len(issues),
            "total_pull_requests": len(prs),
            "total_releases": len(releases),
            "stars": repo_info.get("stargazers_count", 0),
            "forks": repo_info.get("forks_count", 0),
            "language": repo_info.get("language"),
            "created_at": repo_info.get("created_at"),
            "updated_at": repo_info.get("updated_at")
        }

class GitHubWebhookHandler:
    """Handle GitHub webhook events"""
    
    def __init__(self, secret: Optional[str] = None):
        self.secret = secret
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature"""
        if not self.secret:
            return True  # Skip verification if no secret configured
        
        import hmac
        import hashlib
        
        expected = hmac.new(
            self.secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(f"sha256={expected}", signature)
    
    async def handle_push(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle push event"""
        repo = payload.get("repository", {})
        commits = payload.get("commits", [])
        
        logger.info(f"Push event for {repo.get('full_name')}: {len(commits)} commits")
        
        return {
            "event": "push",
            "repository": repo.get("full_name"),
            "branch": payload.get("ref", "").replace("refs/heads/", ""),
            "commits": len(commits),
            "pusher": payload.get("pusher", {}).get("name")
        }
    
    async def handle_pull_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pull request event"""
        action = payload.get("action")
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {})
        
        logger.info(f"PR {action} for {repo.get('full_name')}: #{pr.get('number')}")
        
        return {
            "event": "pull_request",
            "action": action,
            "repository": repo.get("full_name"),
            "pr_number": pr.get("number"),
            "pr_title": pr.get("title"),
            "author": pr.get("user", {}).get("login")
        }
    
    async def handle_issues(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle issues event"""
        action = payload.get("action")
        issue = payload.get("issue", {})
        repo = payload.get("repository", {})
        
        logger.info(f"Issue {action} for {repo.get('full_name')}: #{issue.get('number')}")
        
        return {
            "event": "issues",
            "action": action,
            "repository": repo.get("full_name"),
            "issue_number": issue.get("number"),
            "issue_title": issue.get("title"),
            "author": issue.get("user", {}).get("login")
        }

# Custom exceptions
class GitHubClientError(Exception):
    """Base GitHub client error"""
    pass

class GitHubAuthError(GitHubClientError):
    """GitHub authentication error"""
    pass

class GitHubRateLimitError(GitHubClientError):
    """GitHub rate limit error"""
    pass

class GitHubNotFoundError(GitHubClientError):
    """GitHub resource not found error"""
    pass

# Integration with HE-Graph-Embeddings
class HEGraphGitHubIntegration:
    """Integration between HE-Graph-Embeddings and GitHub"""
    
    def __init__(self, github_client: GitHubClient):
        self.github = github_client
    
    async def create_experiment_issue(self, owner: str, repo: str,
                                    experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create issue with experiment results"""
        title = f"HE-Graph Experiment Results - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
# HE-Graph-Embeddings Experiment Results

## Configuration
- **Model**: {experiment_results.get('model_type', 'Unknown')}
- **Security Level**: {experiment_results.get('security_level', 'Unknown')} bits
- **Dataset**: {experiment_results.get('dataset', 'Unknown')}

## Performance Metrics
- **Encryption Time**: {experiment_results.get('encryption_time_ms', 0):.2f} ms
- **Inference Time**: {experiment_results.get('inference_time_ms', 0):.2f} ms
- **Noise Budget**: {experiment_results.get('final_noise_budget', 0):.2f}
- **Accuracy**: {experiment_results.get('accuracy', 0):.4f}

## System Information
- **GPU**: {experiment_results.get('gpu_name', 'CPU')}
- **Memory Usage**: {experiment_results.get('memory_mb', 0):.2f} MB
- **Timestamp**: {experiment_results.get('timestamp', datetime.now().isoformat())}

Generated automatically by HE-Graph-Embeddings
        """
        
        return await self.github.create_issue(
            owner, repo, title, body,
            labels=["experiment", "automated", "he-graph"]
        )
    
    async def create_performance_pr(self, owner: str, repo: str,
                                   branch: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create pull request with performance improvements"""
        title = f"Performance optimization: {performance_data.get('improvement_type', 'General')}"
        
        body = f"""
## Performance Optimization

### Changes
{performance_data.get('description', 'Performance improvements')}

### Benchmark Results
- **Before**: {performance_data.get('before_time_ms', 0):.2f} ms
- **After**: {performance_data.get('after_time_ms', 0):.2f} ms
- **Improvement**: {performance_data.get('improvement_percent', 0):.1f}%

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks improved
- [ ] Security tests pass

### Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Benchmarks included

Generated by HE-Graph-Embeddings automated optimization
        """
        
        return await self.github.create_pull_request(
            owner, repo, title, body, branch
        )