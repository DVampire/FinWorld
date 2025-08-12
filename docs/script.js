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
        console.log(`Loading table: ${tableId} from ${tablePath}`);
        
        const container = document.getElementById(tableId);
        if (!container) {
            console.error(`Container not found: ${tableId}`);
            return;
        }
        
        // 添加加载状态
        container.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #666;">
                <div style="width: 40px; height: 40px; border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 20px;"></div>
                <p>Loading table...</p>
            </div>
        `;
        
        const response = await fetch(tablePath);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const html = await response.text();
        console.log(`Table content loaded for ${tableId}:`, html.substring(0, 200) + '...');
        
        if (html.trim() === '') {
            throw new Error('Empty table content');
        }
        
        container.innerHTML = html;
        console.log(`Table ${tableId} loaded successfully`);
        
        // 触发表格动画
        setTimeout(() => {
            const tableRows = container.querySelectorAll('.table-row');
            tableRows.forEach((row, index) => {
                setTimeout(() => {
                    row.classList.add('animate');
                }, index * 100);
            });
        }, 100);
        
    } catch (error) {
        console.error(`Error loading table ${tableId}:`, error);
        
        const container = document.getElementById(tableId);
        if (container) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #666; border: 2px dashed #ddd; border-radius: 8px;">
                    <i class="fas fa-exclamation-triangle" style="font-size: 2rem; color: #f59e0b; margin-bottom: 10px;"></i>
                    <h3>Table Loading Error</h3>
                    <p>Failed to load table: ${tableId}</p>
                    <p style="font-size: 0.9rem; color: #999;">Error: ${error.message}</p>
                    <button onclick="loadTable('${tableId}', '${tablePath}')" style="margin-top: 10px; padding: 8px 16px; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer;">
                        <i class="fas fa-redo"></i> Retry
                    </button>
                </div>
            `;
        }
    }
}

// Load all tables and initialize animations when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, starting table loading...');
    
    // Load tables with delay to ensure proper loading
    setTimeout(() => {
        loadTable('forecasting-table', 'assets/table_forecasting_simple.html');
    }, 100);
    
    setTimeout(() => {
        loadTable('trading-table', 'assets/table_trading.html');
    }, 200);
    
    setTimeout(() => {
        loadTable('portfolio-table', 'assets/table_portfolio.html');
    }, 300);
    
    setTimeout(() => {
        loadTable('platform-table', 'assets/table_platform_simple.html');
    }, 400);
    
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

// 基于example目录的动态效果增强

// 页面加载动画处理
window.addEventListener('load', function() {
    setTimeout(() => {
        const loader = document.getElementById('pageLoader');
        if (loader) {
            loader.style.opacity = '0';
            loader.style.visibility = 'hidden';
            document.body.classList.add('loaded');
        }
    }, 1500);
});

// 如果页面已经加载完成，立即隐藏加载器
if (document.readyState === 'complete') {
    const loader = document.getElementById('pageLoader');
    if (loader) {
        loader.style.opacity = '0';
        loader.style.visibility = 'hidden';
    }
}

// 鼠标跟随效果
function createMouseFollower() {
    // 创建鼠标跟随元素
    const follower = document.createElement('div');
    follower.id = 'mouseFollower';
    follower.style.cssText = `
        position: fixed;
        width: 20px;
        height: 20px;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.6) 0%, rgba(102, 126, 234, 0.2) 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 9998;
        transition: transform 0.1s ease, width 0.3s ease, height 0.3s ease;
        mix-blend-mode: screen;
    `;
    document.body.appendChild(follower);
    
    document.addEventListener('mousemove', (e) => {
        follower.style.transform = `translate(${e.clientX - 10}px, ${e.clientY - 10}px)`;
    });

    // 悬停效果
    const hoverElements = document.querySelectorAll('a, button, .contribution-card, .feature-item, .layer, .dataset-card');
    hoverElements.forEach(element => {
        element.addEventListener('mouseenter', () => {
            follower.style.width = '40px';
            follower.style.height = '40px';
            follower.style.background = 'radial-gradient(circle, rgba(102, 126, 234, 0.8) 0%, rgba(102, 126, 234, 0.3) 70%)';
        });
        element.addEventListener('mouseleave', () => {
            follower.style.width = '20px';
            follower.style.height = '20px';
            follower.style.background = 'radial-gradient(circle, rgba(102, 126, 234, 0.6) 0%, rgba(102, 126, 234, 0.2) 70%)';
        });
    });
}

// 粒子背景效果
function createParticles() {
    const hero = document.querySelector('.hero');
    if (!hero) return;
    
    const container = document.createElement('div');
    container.id = 'particlesBg';
    container.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        z-index: 1;
    `;
    
    const particleCount = 50;
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.cssText = `
            position: absolute;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            animation: float-particle 6s ease-in-out infinite;
        `;
        
        // 随机大小和位置
        const size = Math.random() * 4 + 2;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.top = `${Math.random() * 100}%`;
        particle.style.animationDelay = `${Math.random() * 6}s`;
        particle.style.animationDuration = `${Math.random() * 3 + 3}s`;
        
        container.appendChild(particle);
    }
    
    hero.style.position = 'relative';
    hero.appendChild(container);
}

// 卡片放大效果
function add3DCardEffects() {
    const cards = document.querySelectorAll('.contribution-card, .feature-item, .dataset-card');
    
    cards.forEach(card => {
        card.style.transition = 'transform 0.3s ease, box-shadow 0.3s ease';
        
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.05)';
            this.style.boxShadow = '0 10px 25px rgba(0, 0, 0, 0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.boxShadow = '';
        });
    });
}

// 渐变文字效果
function addGradientTextEffects() {
    const titles = document.querySelectorAll('.section-title, .hero-content h1, .hero-content h2');
    
    titles.forEach(title => {
        title.style.background = 'linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #667eea)';
        title.style.backgroundSize = '400% 400%';
        title.style.webkitBackgroundClip = 'text';
        title.style.webkitTextFillColor = 'transparent';
        title.style.backgroundClip = 'text';
        title.style.animation = 'gradient 3s ease infinite';
    });
}

// 按钮发光效果
function addGlowButtonEffects() {
    const buttons = document.querySelectorAll('.btn');
    
    buttons.forEach(button => {
        button.style.position = 'relative';
        button.style.overflow = 'hidden';
        button.style.transition = 'all 0.3s ease';
        
        // 添加调试信息
        console.log('Button found:', button.textContent.trim(), button.href);
        
        // 添加发光效果
        button.addEventListener('mouseenter', function() {
            this.style.boxShadow = '0 0 20px rgba(102, 126, 234, 0.5)';
            this.style.transform = 'translateY(-2px)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.boxShadow = '';
            this.style.transform = 'translateY(0)';
        });
        
        // 添加波纹效果
        button.addEventListener('click', function(e) {
            console.log('Button clicked:', this.textContent.trim(), this.href);
            
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                transform: scale(0);
                animation: ripple 0.6s linear;
                pointer-events: none;
            `;
            
            this.appendChild(ripple);
            setTimeout(() => ripple.remove(), 600);
        });
    });
}

// 打字机效果增强
function enhanceTypewriterEffect() {
    const heroTitle = document.querySelector('.hero-content h1');
    if (heroTitle) {
        const text = heroTitle.textContent;
        heroTitle.textContent = '';
        heroTitle.style.borderRight = '2px solid #667eea';
        heroTitle.style.animation = 'blink 1s infinite';
        
        let i = 0;
        function typeWriter() {
            if (i < text.length) {
                heroTitle.textContent += text.charAt(i);
                i++;
                setTimeout(typeWriter, 150);
            } else {
                setTimeout(() => {
                    heroTitle.style.borderRight = 'none';
                    heroTitle.style.animation = 'none';
                }, 1000);
            }
        }
        
        setTimeout(typeWriter, 1000);
    }
}

// 滚动动画增强
function enhanceScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // 为不同方向的元素添加不同的动画类
    const leftElements = document.querySelectorAll('.contribution-card:nth-child(odd), .feature-item:nth-child(odd)');
    const rightElements = document.querySelectorAll('.contribution-card:nth-child(even), .feature-item:nth-child(even)');
    
    leftElements.forEach(el => {
        el.classList.add('scroll-animate-left');
        el.style.opacity = '0';
        el.style.transform = 'translateX(-50px)';
        observer.observe(el);
    });
    
    rightElements.forEach(el => {
        el.classList.add('scroll-animate-right');
        el.style.opacity = '0';
        el.style.transform = 'translateX(50px)';
        observer.observe(el);
    });
}

// 数字计数动画增强
function enhanceCounterAnimations() {
    const counters = document.querySelectorAll('.text-2xl, .text-3xl, .text-4xl');
    
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const counter = entry.target;
                const text = counter.textContent;
                const numbers = text.match(/\d+/g);
                
                if (numbers) {
                    numbers.forEach(num => {
                        const target = parseInt(num);
                        let current = 0;
                        const duration = 2000;
                        const step = target / (duration / 16);
                        
                        const updateCounter = () => {
                            current += step;
                            if (current < target) {
                                counter.textContent = counter.textContent.replace(num, Math.floor(current));
                                requestAnimationFrame(updateCounter);
                            } else {
                                counter.textContent = counter.textContent.replace(num, target);
                            }
                        };
                        updateCounter();
                    });
                }
            }
        });
    }, { threshold: 0.5 });
    
    counters.forEach(counter => counterObserver.observe(counter));
}

// 添加CSS动画样式
const enhancedStyles = document.createElement('style');
enhancedStyles.textContent = `
    @keyframes float-particle {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    @keyframes gradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes blink {
        0%, 50% { border-color: transparent; }
        51%, 100% { border-color: #667eea; }
    }
    
    @keyframes ripple {
        to { transform: scale(4); opacity: 0; }
    }
    
    .scroll-animate-left {
        transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .scroll-animate-left.animate {
        opacity: 1;
        transform: translateX(0);
    }
    
    .scroll-animate-right {
        transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .scroll-animate-right.animate {
        opacity: 1;
        transform: translateX(0);
    }
    
    .contribution-card, .feature-item, .layer, .dataset-card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .contribution-card:hover, .feature-item:hover, .layer:hover, .dataset-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    .btn {
        transition: all 0.3s ease;
    }
    
    .btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .section-title {
        transition: all 0.3s ease;
    }
    
    .section-title:hover {
        transform: scale(1.05);
    }
    
    .hero-content h1, .hero-content h2 {
        transition: all 0.3s ease;
    }
    
    .hero-content h1:hover, .hero-content h2:hover {
        transform: scale(1.02);
    }
    
    .layer-number {
        transition: all 0.3s ease;
    }
    
    .layer-number:hover {
        transform: scale(1.2) rotate(360deg);
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
    
    .feature-item i {
        transition: all 0.3s ease;
    }
    
    .feature-item:hover i {
        transform: scale(1.3) rotate(10deg);
        color: #667eea;
    }
    
    .dataset-card img {
        transition: all 0.3s ease;
    }
    
    .dataset-card:hover img {
        transform: scale(1.1);
        filter: brightness(1.1);
    }
    
    /* 页面加载动画 */
    .page-loader {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        transition: opacity 0.8s ease-out, visibility 0.8s ease-out;
    }
    
    .loader-content {
        text-align: center;
        color: white;
    }
    
    .loader-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid rgba(255,255,255,0.3);
        border-top: 4px solid white;
        border-radius: 50%;
        animation: spin 1.2s linear infinite;
        margin: 0 auto 20px;
    }
    
    .loader-text {
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 10px;
    }
    
    .loader-subtext {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;

document.head.appendChild(enhancedStyles);

// 初始化所有增强的动态效果
document.addEventListener('DOMContentLoaded', () => {
    // 创建页面加载器
    const loader = document.createElement('div');
    loader.id = 'pageLoader';
    loader.className = 'page-loader';
    loader.innerHTML = `
        <div class="loader-content">
            <div class="loader-spinner"></div>
            <div class="loader-text">FinWorld</div>
            <div class="loader-subtext">Loading...</div>
        </div>
    `;
    document.body.appendChild(loader);
    
    // 初始化所有动态效果
    createMouseFollower();
    createParticles();
    add3DCardEffects();
    addGradientTextEffects();
    addGlowButtonEffects();
    enhanceTypewriterEffect();
    enhanceScrollAnimations();
    enhanceCounterAnimations();
    
    // 确保按钮点击功能正常
    const allButtons = document.querySelectorAll('.btn, a[href]');
    allButtons.forEach(button => {
        // 确保按钮可以点击
        button.style.pointerEvents = 'auto';
        button.style.cursor = 'pointer';
        
        // 添加点击事件监听器确保链接正常工作
        if (button.href && !button.href.includes('#')) {
            button.addEventListener('click', function(e) {
                console.log('Button clicked, navigating to:', this.href);
                // 不阻止默认行为，让链接正常工作
            });
        }
    });
    
    // 延迟隐藏加载器
    setTimeout(() => {
        if (loader) {
            loader.style.opacity = '0';
            loader.style.visibility = 'hidden';
        }
    }, 1500);
});
