// Mobile Navigation Toggle
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('active');
    navMenu.classList.toggle('active');
});

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-menu a').forEach(link => {
    link.addEventListener('click', () => {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');
    });
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Navbar background change on scroll
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 100) {
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.boxShadow = 'none';
    }
});

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Typing effect for hero title
function typeWriter(element, text, speed = 100) {
    let i = 0;
    element.innerHTML = '';
    
    function type() {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    
    type();
}

// Initialize typing effect when page loads
window.addEventListener('load', () => {
    const heroTitle = document.querySelector('.hero-content h1');
    if (heroTitle) {
        const originalText = heroTitle.textContent;
        typeWriter(heroTitle, originalText, 150);
    }
});

// Parallax effect for hero section
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero');
    const heroImage = document.querySelector('.hero-image');
    
    if (hero && heroImage) {
        const rate = scrolled * -0.5;
        heroImage.style.transform = `translateY(${rate}px)`;
    }
});

// Add loading animation
window.addEventListener('load', () => {
    document.body.classList.add('loaded');
});

// Copy code blocks functionality
document.addEventListener('DOMContentLoaded', () => {
    const codeBlocks = document.querySelectorAll('.code-block');
    
    codeBlocks.forEach(block => {
        const copyButton = document.createElement('button');
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.className = 'copy-btn';
        copyButton.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            padding: 8px;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.3s ease;
        `;
        
        copyButton.addEventListener('click', () => {
            const code = block.querySelector('code').textContent;
            navigator.clipboard.writeText(code).then(() => {
                copyButton.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                }, 2000);
            });
        });
        
        block.style.position = 'relative';
        block.appendChild(copyButton);
    });
});

// Add hover effects for buttons
document.addEventListener('DOMContentLoaded', () => {
    const buttons = document.querySelectorAll('.btn');
    
    buttons.forEach(button => {
        button.addEventListener('mouseenter', () => {
            button.style.transform = 'translateY(-2px) scale(1.05)';
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.transform = 'translateY(0) scale(1)';
        });
    });
});

// Add scroll progress indicator
window.addEventListener('scroll', () => {
    const scrollTop = window.pageYOffset;
    const docHeight = document.body.offsetHeight - window.innerHeight;
    const scrollPercent = (scrollTop / docHeight) * 100;
    
    // Create progress bar if it doesn't exist
    let progressBar = document.querySelector('.scroll-progress');
    if (!progressBar) {
        progressBar = document.createElement('div');
        progressBar.className = 'scroll-progress';
        progressBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 0%;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            z-index: 9999;
            transition: width 0.3s ease;
        `;
        document.body.appendChild(progressBar);
    }
    
    progressBar.style.width = scrollPercent + '%';
});

// Load HTML tables
async function loadTable(tableId, tablePath) {
    try {
        const response = await fetch(tablePath);
        const html = await response.text();
        const container = document.getElementById(tableId);
        if (container) {
            container.innerHTML = html;
        }
    } catch (error) {
        console.error(`Error loading table ${tableId}:`, error);
    }
}

// Load all tables and initialize animations when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Load tables
    loadTable('forecasting-table', 'assets/table_forecasting.html');
    loadTable('trading-table', 'assets/table_trading.html');
    loadTable('portfolio-table', 'assets/table_portfolio.html');
    loadTable('platform-table', 'assets/table_platform.html');
    
    // Initialize animations for all elements
    const animatedElements = document.querySelectorAll('.contribution-card, .feature-item, .layer, .result-card, .dataset-card');
    
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// 表格模态框功能
function showTableModal(tableId, title) {
    // 创建模态框
    const modalOverlay = document.createElement('div');
    modalOverlay.className = 'modal-overlay';
    modalOverlay.id = 'table-modal';
    
    // 获取原始表格内容
    const originalElement = document.getElementById(tableId);
    const elementContent = originalElement.outerHTML;
    
    // 创建模态框内容
    modalOverlay.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">${title}</h2>
                <button class="modal-close" onclick="closeTableModal()">&times;</button>
            </div>
            <div class="modal-table-container">
                ${elementContent}
            </div>
        </div>
    `;
    
    // 添加到页面
    document.body.appendChild(modalOverlay);
    
    // 显示模态框
    setTimeout(() => {
        modalOverlay.classList.add('active');
    }, 10);
    
    // 处理表格显示
    const completeGrid = modalOverlay.querySelector('.complete-table-grid');
    if (completeGrid) {
        completeGrid.style.display = 'flex';
        
        // 处理所有表格
        const tables = completeGrid.querySelectorAll('table');
        tables.forEach(table => {
            table.classList.remove('thumbnail-table', 'complete-table');
            table.classList.add('modal-table');
            table.style.display = 'table';
        });
    }
}

function closeTableModal() {
    const modal = document.getElementById('table-modal');
    if (modal) {
        modal.classList.remove('active');
        setTimeout(() => {
            modal.remove();
        }, 300);
    }
}

// 点击模态框外部关闭
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-overlay')) {
        closeTableModal();
    }
});
