// Navigation HTML
const navigationHTML = `
<nav class="navbar">
    <div class="nav-container">
        <div class="nav-logo">
            <h2>FinWorld</h2>
        </div>
        <ul class="nav-menu">
            <li><a href="index.html">Home</a></li>
            <li><a href="index.html#overview">Overview</a></li>
            <li><a href="index.html#architecture">Architecture</a></li>
            <li><a href="index.html#datasets">Datasets</a></li>
            <li><a href="index.html#results">Results</a></li>
            <li><a href="index.html#download">Download</a></li>
            <li><a href="documentation.html">Documentation</a></li>
        </ul>
        <div class="hamburger">
            <span></span>
            <span></span>
            <span></span>
        </div>
    </div>
</nav>
`;

// Documentation Sidebar HTML
const documentationSidebarHTML = `
<aside class="docs-sidebar" id="docs-sidebar">
    <div class="sidebar-toggle" id="sidebar-toggle">
        <i class="fas fa-bars"></i>
    </div>
    <div class="sidebar-content">
        <div class="sidebar-section">
            <h3>Get Started</h3>
            <ul>
                <li><a href="documentation.html#introduction">Introduction</a></li>
                <li><a href="documentation.html#installation">Installation</a></li>
                <li><a href="documentation.html#quick-start">Quick Start</a></li>
            </ul>
        </div>
        
        <div class="sidebar-section">
            <h3>Core Documentation</h3>
            <ul>
                <li><a href="architecture.html">Architecture Guide</a></li>
                <li><a href="tasks.html">Financial Tasks</a></li>
                <li><a href="configuration.html">Configuration Guide</a></li>
                <li><a href="api-reference.html">API Reference</a></li>
                <li><a href="tutorials.html">Tutorials</a></li>
                <li><a href="examples.html">Examples</a></li>
            </ul>
        </div>
        
        <div class="sidebar-section">
            <h3>Advanced Topics</h3>
            <ul>
                <li><a href="#custom-models">Custom Models</a></li>
            </ul>
        </div>
    </div>
</aside>
`;

// Footer HTML
const footerHTML = `
<footer class="footer">
    <div class="footer-content" style="text-align: center; max-width: 800px; margin: 0 auto;">
        <div class="footer-section">
            <h3>Citation</h3>
            <p style="text-align: left; font-family: monospace; font-size: 14px; line-height: 1.6;">@article{zhang2025finworld,<br>
            &nbsp;&nbsp;title={FinWorld: An All-in-One Open-Source Platform for End-to-End Financial AI Research and Deployment},<br>
            &nbsp;&nbsp;author={Zhang, Wentao and Zhao, Yilei and Zong, Chuqiao and Wang, Xinrun and An, Bo},<br>
            &nbsp;&nbsp;journal={arXiv preprint arXiv:2508.02292},<br>
            &nbsp;&nbsp;year={2025}<br>
            }</p>
        </div>
    </div>
    <div class="footer-bottom" style="text-align: center;">
        <p>&copy; 2025 FinWorld Team. Built with ❤️ for the financial AI community.</p>
    </div>
</footer>
`;

// Function to get GitHub base URL
function getGitHubBaseURL() {
    // Check if we're on GitHub Pages
    if (window.location.hostname.includes('github.io')) {
        // Extract repository name from the path
        const pathParts = window.location.pathname.split('/');
        if (pathParts.length >= 3) {
            const username = pathParts[1];
            const repoName = pathParts[2];
            return `https://github.com/${username}/${repoName}`;
        }
    }
    
    // Default to FinWorld repository
    return 'https://github.com/DVampire/FinWorld';
}

// Function to convert relative paths to absolute GitHub URLs
function convertToGitHubURLs() {
    const githubBase = getGitHubBaseURL();
    
    // Find all links with relative paths starting with ../
    const links = document.querySelectorAll('a[href^="../"]');
    links.forEach(link => {
        const relativePath = link.getAttribute('href');
        // Remove the ../ prefix and convert to absolute GitHub URL
        const cleanPath = relativePath.replace('../', '');
        
        // Determine if it's a file or directory based on extension
        const isFile = cleanPath.includes('.py') || cleanPath.includes('.yaml') || cleanPath.includes('.json') || cleanPath.includes('.md');
        const urlType = isFile ? 'blob' : 'tree';
        
        const absoluteURL = `${githubBase}/${urlType}/main/${cleanPath}`;
        link.setAttribute('href', absoluteURL);
    });
}

// Inject components
document.addEventListener('DOMContentLoaded', function() {
    // Inject navigation
    const navContainer = document.getElementById('navigation-container');
    if (navContainer) {
        navContainer.innerHTML = navigationHTML;
    }

    // Inject documentation sidebar
    const sidebarContainer = document.getElementById('sidebar-container');
    if (sidebarContainer) {
        sidebarContainer.innerHTML = documentationSidebarHTML;
        
        // Set active page based on current URL
        const currentPage = window.location.pathname.split('/').pop();
        const activeLink = sidebarContainer.querySelector(`a[href="${currentPage}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
        
        // Initialize sidebar toggle after injection
        setTimeout(() => {
            initializeSidebarToggle();
        }, 50);
    }

    // Inject footer
    const footerContainer = document.getElementById('footer-container');
    if (footerContainer) {
        footerContainer.innerHTML = footerHTML;
    }

    // Convert relative GitHub links to absolute URLs
    setTimeout(convertToGitHubURLs, 100);

    // Initialize mobile menu after components are injected
    setTimeout(initializeMobileMenu, 100);
});

function initializeMobileMenu() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');

    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            navMenu.classList.toggle('active');
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!hamburger.contains(e.target) && !navMenu.contains(e.target)) {
                navMenu.classList.remove('active');
            }
        });
    }
}

function initializeSidebarToggle() {
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('docs-sidebar');
    const docsContent = document.querySelector('.docs-content');

    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            sidebar.classList.toggle('collapsed');
            if (docsContent) {
                docsContent.classList.toggle('sidebar-collapsed');
            }
        });
    }
}
