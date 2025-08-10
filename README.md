# FinWorld: An All-in-One Open-Source Platform for End-to-End Financial AI Research and Deployment

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2508.02292-b31b1b.svg)](https://arxiv.org/abs/2508.02292)
[![Website](https://img.shields.io/badge/Website-FinWorld-blue.svg)](https://dvampire.github.io/FinWorld/)

**FinWorld** is a comprehensive, all-in-one open-source platform that provides end-to-end support for the entire financial AI workflow, from data acquisition to experimentation and deployment. Built on the foundation of unified AI paradigms, heterogeneous data integration, and advanced agent automation, FinWorld addresses the critical limitations of existing financial AI platforms.

> **Paper**: [FinWorld: An All-in-One Open-Source Platform for End-to-End Financial AI Research and Deployment](https://arxiv.org/abs/2508.02292) (Arxiv)
> **Website**: [FinWorld Project Website](https://dvampire.github.io/FinWorld/) - Interactive demo and detailed results

![FinWorld Architecture](docs/assets/finworld.png)
*Figure 1: Overview of FinWorld's comprehensive architecture and workflow*

## 🎯 Research Contributions

Our main contributions are threefold:

1. **Unified Framework**: We propose a unified, end-to-end framework for training and evaluation of ML, DL, RL, LLMs, and LLM agents, covering four critical financial AI task types including time series forecasting, algorithmic trading, portfolio management, and LLM applications.

2. **Modular Design**: The framework features a modular architecture that enables flexible construction of custom models and tasks, including the development of personalized LLM agents. The system supports efficient distributed training and testing across multiple environments.

3. **Comprehensive Benchmark**: We provide support for multimodal heterogeneous data with over 800 million samples, establishing a comprehensive benchmark for the financial AI community. Extensive experiments across four task types demonstrate the framework's flexibility and effectiveness.

## 🚀 Key Features

### 🔄 Multi-task Support
- **Time Series Forecasting**: Advanced models including Autoformer, Crossformer, DLinear, TimesNet, PatchTST, TimeMixer, and TimeXer
- **Algorithmic Trading**: RL-based (PPO, SAC) and ML/DL-based trading strategies
- **Portfolio Management**: Multi-asset portfolio optimization with various risk measures
- **LLM Applications**: Financial reasoning and LLM agent training with RL fine-tuning

### 📊 Multimodal Data Integration
- **Structured Data**: OHLCV price data, technical indicators, financial factors
- **Unstructured Data**: News articles, financial reports, earnings calls
- **Multi-source Support**: FMP, Alpaca, AKShare, TuShare APIs
- **Market Coverage**: DJ30, SP500, SSE50, HS300 across US and Chinese markets

### 🧠 Comprehensive AI Paradigms
- **Machine Learning**: LightGBM, XGBoost, and traditional ML models
- **Deep Learning**: Transformer, LSTM, and state-of-the-art architectures
- **Reinforcement Learning**: PPO, SAC, and custom RL algorithms
- **Large Language Models**: GPT-4.1, Claude-4-Sonnet, Qwen series, and custom LLMs
- **LLM Agents**: Multi-agent systems with tool use and reasoning capabilities

### 🛠️ Advanced Automation
- **Distributed Training**: Multi-GPU training and testing support
- **Auto Presentation**: Automated report generation and visualization
- **Experiment Tracking**: Integration with WandB and TensorBoard
- **Modular Architecture**: Extensible framework for rapid prototyping

## 📋 Requirements

- Python 3.11+
- CUDA 12.4+ (for GPU acceleration)
- Conda or Miniconda
- 16+ GB RAM (recommended for large datasets)

## 🛠️ Installation

### 1. Create Conda Environment

```bash
conda create -n finworld python=3.11
conda activate finworld
```

### 2. Install Dependencies

```bash
# Install base dependencies
make install-base

# Install browser automation tools
make install-browser

# Install VERL framework
make install-verl
```

### Alternative Installation with Poetry

```bash
# Install Poetry
pip install poetry

# Install dependencies
poetry install
```

## 🚀 Quick Start

### 1. Download Financial Data

```bash
# Download DJ30 data (example)
python scripts/download/download.py --config configs/download/dj30/dj30_fmp_price_1day.py
python scripts/download/download.py --config configs/download/dj30/dj30_fmp_price_1min.py
```

### 2. Train RL Trading Models

```bash
# Train PPO trading models for multiple stocks
CUDA_VISIBLE_DEVICES=0 python scripts/rl_trading/train.py --config=configs/rl_trading/ppo/AAPL_ppo_trading.py
```

### 3. Train Portfolio Models

```bash
# Train PPO portfolio models for different indices
CUDA_VISIBLE_DEVICES=0 python scripts/rl_portfolio/train.py --config=configs/rl_portfolio/ppo/dj30_ppo_portfolio.py
```

### 4. Use Pre-built Scripts

```bash
# Run example scripts
bash examples/ppo_trading.sh
bash examples/ppo_portfolio.sh
bash examples/download.sh
```

## 📊 Empirical Results

### Time Series Forecasting
Our comprehensive evaluation on DJ30 and HS300 datasets demonstrates the superiority of deep learning approaches:

- **TimeXer** achieves MAE of 0.0529 and MSE of 0.0062 on DJ30, significantly outperforming LightGBM (MAE: 0.1392, MSE: 0.0235)
- **TimeMixer** and **TimeXer** show superior performance on HS300 with MAEs of 0.3804 and 0.3727 respectively
- Deep learning models consistently achieve higher RankICIR scores compared to traditional ML methods

### Algorithmic Trading
RL-based methods demonstrate clear advantages in trading performance:

- **SAC** achieves 101.55% ARR on TSLA with superior risk-adjusted returns
- **PPO** attains 2.10 SR on META, outperforming all baseline methods
- RL methods consistently deliver higher returns and better risk metrics across all evaluated stocks

### Portfolio Management
RL-based portfolio optimization shows significant improvements:

- **SAC** achieves up to 31.2% annualized returns on SP500 with Sharpe ratios above 1.5
- RL methods consistently outperform rule-based and ML-based approaches
- Superior risk-adjusted performance across all major indices

### LLM Applications
Our FinReasoner model demonstrates state-of-the-art performance:

- **Financial Reasoning**: Leads all four benchmarks (FinQA, FinEval, ConvFinQA, CFLUE)
- **Trading Capabilities**: Strong performance across all evaluated stocks with comprehensive risk management
- **Domain-specific Training**: Outperforms generic instruction-tuned models

## 🏗️ Architecture Overview

FinWorld employs a layered, object-oriented architecture with seven core layers:

### 1. Configuration Layer
- Built on `mmengine` for unified experiment management
- Registry mechanism for flexible component management
- Support for configuration inheritance and overrides

### 2. Dataset Layer
- **Downloader Module**: Multi-source data acquisition (FMP, Alpaca, AKShare)
- **Processor Module**: Feature engineering and preprocessing (Alpha158 factors)
- **Dataset Module**: Task-specific data organization
- **Environment Module**: RL environment encapsulation

### 3. Model Layer
- **ML Models**: Traditional ML algorithms (LightGBM, XGBoost)
- **DL Models**: Neural architectures (Transformer, LSTM, VAE)
- **RL Models**: Actor-critic networks with financial constraints
- **LLM Models**: Unified interface for commercial and open-source LLMs

### 4. Training Layer
- **Optimizer**: Adam, AdamW, SGD with gradient centralization
- **Loss**: Regression, classification, and RL surrogate losses
- **Scheduler**: Cosine, linear, and adaptive learning rate scheduling
- **Trainer**: Task-specific training pipelines with distributed support

### 5. Evaluation Layer
- **Metrics**: Financial-specific metrics (ARR, SR, MDD, CR, SoR)
- **Visualization**: K-line charts, cumulative returns, compass plots
- **Standardized Protocols**: Consistent evaluation across all tasks

### 6. Task Layer
- **Time Series Forecasting**: Multi-step prediction with various horizons
- **Algorithmic Trading**: Single-asset trading with risk management
- **Portfolio Management**: Multi-asset allocation with constraints
- **LLM Applications**: Financial reasoning and agent training

### 7. Presentation Layer
- **Auto-reporting**: LaTeX technical reports and HTML dashboards
- **Multi-channel Publishing**: GitHub, GitHub Pages, and experiment tracking
- **Version Control**: Systematic archiving and knowledge transfer

## 📁 Project Structure

```
FinWorld/
├── configs/                    # Configuration files
│   ├── _asset_list_/          # Asset list configurations
│   ├── agent/                 # Agent configurations
│   ├── download/              # Data download configurations
│   ├── finreasoner/           # Financial reasoning configs
│   ├── ml_portfolio/          # ML portfolio configurations
│   ├── ml_trading/            # ML trading configurations
│   ├── process/               # Data processing configurations
│   ├── rl_portfolio/          # RL portfolio configurations
│   ├── rl_trading/            # RL trading configurations
│   ├── rule_portfolio/        # Rule-based portfolio configs
│   ├── rule_trading/          # Rule-based trading configs
│   ├── storm/                 # Storm framework configs
│   ├── time/                  # Time series model configs
│   └── vae/                   # VAE model configurations
├── finworld/                  # Core framework
│   ├── agent/                 # Multi-agent system
│   ├── base/                  # Base classes and utilities
│   ├── calendar/              # Calendar management
│   ├── config/                # Configuration management
│   ├── data/                  # Data processing modules
│   ├── diffusion/             # Diffusion models
│   ├── downloader/            # Data downloaders
│   ├── downstream/            # Downstream tasks
│   ├── environment/           # Trading environments
│   ├── evaluator/             # Evaluation metrics
│   ├── exception/             # Exception handling
│   ├── factor/                # Factor models
│   ├── log/                   # Logging system
│   ├── loss/                  # Loss functions
│   ├── memory/                # Memory management
│   ├── metric/                # Performance metrics
│   ├── models/                # AI models and architectures
│   ├── mverl/                 # Multi-agent VERL
│   ├── optimizer/             # Optimization algorithms
│   ├── plot/                  # Visualization tools
│   ├── processor/             # Data processors
│   ├── proxy/                 # Proxy management
│   ├── reducer/               # Dimensionality reduction
│   ├── scheduler/             # Task scheduling
│   ├── task/                  # Task definitions
│   ├── tools/                 # Utility tools and integrations
│   ├── trainer/               # Training frameworks
│   ├── trajectory/            # Trajectory management
│   ├── utils/                 # Utility functions
│   └── verify/                # Verification tools
├── scripts/                   # Training and execution scripts
├── examples/                  # Example usage scripts
├── tests/                     # Unit tests
├── libs/                      # External libraries (VERL)
├── res/                       # Resources and assets
└── tools/                     # Development tools
```

## 🎯 Supported Markets & Data Sources

### Markets
- **US Markets**: DJ30 (Dow Jones 30), SP500 (S&P 500)
- **Chinese Markets**: SSE50 (Shanghai Stock Exchange 50), HS300 (CSI 300)

### Data Sources
- **FMP**: Financial Modeling Prep - Comprehensive financial data
- **Alpaca**: US market data and news feeds
- **AKShare**: Chinese market data and financial information
- **TuShare**: Chinese financial data and research tools

### Data Types
- **Price Data**: OHLCV at daily and minute frequencies
- **News Data**: Financial news and sentiment analysis
- **Technical Indicators**: Alpha158 factors and custom indicators
- **LLM Reasoning**: Financial QA datasets and reasoning benchmarks

## 🔧 Configuration System

FinWorld uses a flexible configuration system based on YAML files and the `mmengine` framework:

- **Experiment Management**: Centralized configuration for reproducibility
- **Component Registry**: Flexible component instantiation and management
- **Inheritance Support**: Configuration inheritance and override mechanisms
- **Validation**: Automatic configuration validation and error checking

## 📊 Performance Metrics

### Trading & Portfolio Metrics
- **ARR**: Annualized Rate of Return
- **SR**: Sharpe Ratio
- **MDD**: Maximum Drawdown
- **CR**: Calmar Ratio
- **SoR**: Sortino Ratio
- **VOL**: Volatility

### Forecasting Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RankIC**: Rank Information Coefficient
- **RankICIR**: Rank Information Coefficient Information Ratio

### LLM Metrics
- **Accuracy**: Task-specific accuracy scores
- **F1 Score**: Precision and recall balance
- **Financial Reasoning**: Domain-specific evaluation metrics

## 🤝 Contributing

We welcome contributions from the research community! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow the existing code style and architecture patterns
- Add comprehensive tests for new functionality
- Update documentation for new features
- Ensure compatibility with existing components

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the following organizations and tools for their contributions:

- **[VERL](https://github.com/volcengine/verl)**: Reinforcement learning framework
- **[FMP](https://financialmodelingprep.com/)**: Financial data API
- **[Alpaca](https://alpaca.markets/)**: Market data provider
- **[AKShare](https://akshare.akfamily.xyz/)**: Chinese financial data
- **[MMEngine](https://github.com/open-mmlab/mmengine)**: Training framework

## 📞 Support & Citation

### Support
For questions, issues, or contributions:
- Open an issue on [GitHub](https://github.com/DVampire/FinWorld)
- Visit our [Project Website](https://dvampire.github.io/FinWorld/) for interactive demo and detailed results
- Check the documentation in the `docs/` directory
- Review example scripts in the `examples/` directory

### Citation
If you find FinWorld useful in your research, please cite our paper:

```bibtex
@article{zhang2025finworld,
  title={FinWorld: An All-in-One Open-Source Platform for End-to-End Financial AI Research and Deployment},
  author={Zhang, Wentao and Zhao, Yilei and Zong, Chuqiao and Wang, Xinrun and An, Bo},
  journal={arXiv preprint arXiv:2508.02292},
  year={2025}
}
```

## 📚 Related Papers

- [TradeMaster: A Unified Trading Platform](https://arxiv.org/abs/2304.10427)
- [Qlib: An AI-oriented Quantitative Investment Platform](https://arxiv.org/abs/2009.11189)
- [FinRL-Meta: A Universe of Near-Real Market Envirionments](https://arxiv.org/abs/2112.06753)

## 🌐 Project Website

Visit our interactive project website for detailed results, visualizations, and comprehensive documentation:

**[https://dvampire.github.io/FinWorld/](https://dvampire.github.io/FinWorld/)**

The website features:
- Interactive architecture overview
- Detailed empirical results with visualizations
- Platform comparison tables
- Quick start guides and examples

---

**FinWorld** - Empowering Financial AI Research and Applications

*Built with ❤️ by the FinWorld Team*