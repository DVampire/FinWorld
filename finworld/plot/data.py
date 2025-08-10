import plotly.graph_objects as go

from finworld.registry import PLOT
from finworld.log import logger

@PLOT.register_module(force=True)
class PlotDownloadData():
    def __init__(self):
        super(PlotDownloadData, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)
    
    def _create_hierarchical_color_scheme(self):
        color_scheme = {
            'US Stock': {
                'base': '#BFCFC0',
                'APIs': {
                    'Alpaca': '#A8C6A8',
                    'FMP': '#C2D6C2'
                },
                'symbols': {
                    'DJ30': '#CDE2CD',
                    'SP500': '#DAEEDA'
                },
                'data_types': {
                    'Price': '#AACCAA',
                    'News': '#96B896'
                },
                'frequencies': {
                    '1d': '#D6EADB',
                    '1m': '#E2F3E7'
                }
            },
            'CN Stock': {
                'base': '#C5DDE1',
                'APIs': {
                    'TuShare': '#A5CED3',
                    'AkShare': '#B7DDE1',
                    'FMP': '#C0E6E9'
                },
                'symbols': {
                    'SSE50': '#C7E9EC',
                    'HS300': '#D8F3F5'
                },
                'data_types': {
                    'Price': '#9BCED3',
                    'News': '#86BCC1'
                },
                'frequencies': {
                    '1d': '#D1EFF2',
                    '1m': '#E4F7F9'
                }
            }
        }

        return color_scheme 
    
    def _format_percentage(self, percentage: float) -> str:
        """
        Format percentage to avoid displaying 0.0%
        Args:
            percentage (float): percentage value

        Returns:
            str: formatted percentage string
        """
        if percentage < 0.01:
            return f"{percentage:.3f}%"  # Display 3 decimal places
        elif percentage < 0.1:
            return f"{percentage:.2f}%"  # Display 2 decimal places
        else:
            return f"{percentage:.1f}%"  # Display 1 decimal place
        
    def _summary(self, data_dict: dict):
        """
        Summary of data_dict
        Args:
            data_dict (dict): data_dict
        """
        total_count = sum(
            count for market in data_dict.values()
            for api in market.values()
            for symbol in api.values()
            for dtype in symbol.values()
            for freq, (count, _, _) in dtype.items()
        )

        logger.info("Data statistics summary:")
        logger.info(f"Total records: {total_count:,}")

        logger.info("\nDistribution of records by market:")
        for market, apis in data_dict.items():
            market_count = sum(
                count for api in apis.values()
                for symbol in api.values()
                for dtype in symbol.values()
                for freq, (count, _, _) in dtype.items()
            )
            logger.info(f"{market}: {market_count:,} ({market_count/total_count*100:.1f}%)")

        logger.info("\nDistribution of records by API:")
        api_counts = {}
        for market, apis in data_dict.items():
            for api, symbols in apis.items():
                api_count = sum(
                    count for symbol in symbols.values()
                    for dtype in symbol.values()
                    for freq, (count, _, _) in dtype.items()
                )
                api_counts[api] = api_count

        for api, count in sorted(api_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{api}: {count:,} ({count/total_count*100:.1f}%)")

    def plot(self, 
             data_dict: dict,
             savefig: str = 'download_data.pdf',
             **kwargs):
        
        color_scheme = self._create_hierarchical_color_scheme()
    
        # Initialize variables for hierarchical data
        labels = []
        parents = []
        values = []
        colors = []
        ids = []
        
        # Initialize dictionary to store cumulative values for each level
        level_values = {}
        
        # Calculate total count
        total_count = sum(
            count for market in data_dict.values()
            for api in market.values()
            for symbol in api.values()
            for dtype in symbol.values()
            for freq, (count, _, _) in dtype.items()
        )
        
        # Set minimum percentage threshold (e.g. 1%)
        min_percentage_threshold = 10
        min_count_threshold = (min_percentage_threshold / 100) * total_count
        
        # Iterate through data to build hierarchical structure
        for market, apis in data_dict.items():
            for api, symbols in apis.items():
                for symbol, types in symbols.items():
                    for dtype, freqs in types.items():
                        for freq, (count, start_date, end_date) in freqs.items():
                            
                            # Calculate percentage
                            percentage = (count / total_count) * 100
                            
                            # If percentage is less than threshold, use minimum threshold value
                            display_count = max(count, min_count_threshold)
                            display_percentage = max(percentage, min_percentage_threshold)
                            
                            # Level 1: Market
                            if market not in level_values:
                                level_values[market] = 0
                            level_values[market] += display_count  # Use display value
                            
                            if market not in labels:
                                labels.append(market)  # Add first, then update
                                parents.append('')
                                values.append(0)  # Set to 0 first, then update
                                ids.append(market)
                                colors.append(color_scheme[market]['base'])
                            
                            # Level 2: API
                            api_id = f"{market}>{api}"
                            if api_id not in level_values:
                                level_values[api_id] = 0
                            level_values[api_id] += display_count  # Use display value
                            
                            if api_id not in ids:
                                labels.append(api)  # Add first, then update
                                parents.append(market)
                                values.append(0)  # Set to 0 first, then update
                                ids.append(api_id)
                                colors.append(color_scheme[market]['APIs'][api])
                            
                            # Level 3: Symbol
                            symbol_id = f"{market}>{api}>{symbol}"
                            if symbol_id not in level_values:
                                level_values[symbol_id] = 0
                            level_values[symbol_id] += display_count  # Use display value
                            
                            if symbol_id not in ids:
                                labels.append(symbol)  # Add first, then update
                                parents.append(api_id)
                                values.append(0)  # Set to 0 first, then update
                                ids.append(symbol_id)
                                colors.append(color_scheme[market]['symbols'][symbol])
                            
                            # Level 4: Data Type + Time (outermost level)
                            # Calculate percentage based on display_count (for pie chart size)
                            # But display real percentage in labels
                            real_percentage = (count / total_count) * 100
                            time_label = f"{dtype}<br>{freq}<br>{start_date[:7]}~{end_date[:7]}<br>{self._format_percentage(real_percentage)}"  # Display real percentage
                            dtype_id = f"{market}>{api}>{symbol}>{time_label}"
                            labels.append(time_label)
                            parents.append(symbol_id)
                            values.append(display_count)  # Use display value (for pie chart size)
                            ids.append(dtype_id)
                            colors.append(color_scheme[market]['frequencies'][freq])
        
        # Update values and labels for intermediate levels
        for i, id_val in enumerate(ids):
            if id_val in level_values:
                values[i] = level_values[id_val]
                # Add percentage information for intermediate levels
                if id_val in ['US Stock', 'CN Stock']:  # Market level
                    # Calculate real market total value
                    real_market_total = sum(
                        count for market_name, apis in data_dict.items()
                        for api in apis.values()
                        for symbol in api.values()
                        for dtype in symbol.values()
                        for freq, (count, _, _) in dtype.items()
                        if market_name == id_val
                    )
                    percentage = (real_market_total / total_count) * 100
                    labels[i] = f"{labels[i]}<br>{self._format_percentage(percentage)}"
                elif '>' in id_val and id_val.count('>') == 1:  # API level
                    # Calculate real API total value
                    api_name = id_val.split('>')[1]
                    market_name = id_val.split('>')[0]
                    real_api_total = sum(
                        count for symbol in data_dict[market_name][api_name].values()
                        for dtype in symbol.values()
                        for freq, (count, _, _) in dtype.items()
                    )
                    percentage = (real_api_total / total_count) * 100
                    labels[i] = f"{labels[i]}<br>{self._format_percentage(percentage)}"
                elif '>' in id_val and id_val.count('>') == 2:  # Symbol level
                    # Calculate real symbol total value
                    parts = id_val.split('>')
                    market_name, api_name, symbol_name = parts[0], parts[1], parts[2]
                    real_symbol_total = sum(
                        count for dtype in data_dict[market_name][api_name][symbol_name].values()
                        for freq, (count, _, _) in dtype.items()
                    )
                    percentage = (real_symbol_total / total_count) * 100
                    labels[i] = f"{labels[i]}<br>{self._format_percentage(percentage)}"
        
        # Create figure
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            ids=ids,
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>' +
                        'Number: %{value:,}<br>' +
                        'Percentage of parent: %{percentParent}<br>' +
                        'Percentage of total: %{percentRoot}<extra></extra>',
            marker=dict(
                colors=colors,
                line=dict(color="#FFFFFF", width=1)
            ),
            textfont=dict(size=22),
            insidetextorientation='radial',  # Change to radial, make outermost text vertical to tangent
            textinfo='label',  # Display label text
            rotation=90
        ))
        
        # Set layout
        fig.update_layout(
            title={
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial Black', 'color': '#2c3e50'}
            },
            font=dict(size=10, family="Arial"),
            margin=dict(t=100, l=40, r=40, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white',
            width=1000,
            height=800
        )
        
        # Save as high-quality image
        fig.write_image(
            savefig,
            width=1200,
            height=900,
            scale=2,  # Increase resolution
            engine="kaleido"
        )
        
        self._summary(data_dict)
        
        return fig
    
    
@PLOT.register_module(force=True)
class PlotLLMData():
    def __init__(self):
        super(PlotLLMData, self).__init__()
        
    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)
    
    def _create_hierarchical_color_scheme(self):
        """
        Create hierarchical color scheme, each level has unique color but keep gradient color scheme
        """
        color_scheme = {
            # EN blue gradient
            'EN': {
                'base': '#C9DDF0',  # Main background gray blue
                'certifications': {
                    'ACCA': '#BFD6EC',  # Darker
                    'FINANCE': '#C9DDF0',  # Base gray blue
                    'CFA': '#D7E5F4',  # Light gray blue
                    'CRA': '#E3EDF7',  # Darker gray blue
                    'FRM': '#EEF4FA',  # Very light gray blue
                    'ESG': '#D0E1EE',  # Medium gray
                    'TR': '#BED2E5',  # Gray but slightly darker, level change
                    'SM': '#A5BEDA'  # Grayer, colder
                }
            },
            'CN': {
                'base': '#FFE7AE',  # Soft light yellow, inspired by Training block
                'certifications': {
                    'AMAC': '#FFE7AE',
                    'AFP': '#FFEDC4',
                    'EXAM': '#FFF3D9',
                    'CCBP_I': '#FFF9EE',
                    'CCBP_P': '#FFFAF4',
                    'CFQ': '#FFFDF8',
                    'CSP': '#FFF9E3',
                    'CCA': '#FFF6D8',
                    'CPA': '#FFF2C8',
                    'CIC': '#FFEEC0',
                    'CCDE': '#FFEBB8',
                    'GTQC': '#FFE4A5',
                    'IEPQ': '#FFE1A0',
                    'JEPQ': '#FFD991',
                    'JAPQ': '#FFD481',
                    'SSE': '#FFCE70'
                }
            }
        }
        return color_scheme
    
    def _format_percentage(self, percentage: float) -> str:
        """
        Format percentage to avoid displaying 0.0%
        Args:
            percentage (float): percentage value

        Returns:
            str: formatted percentage string
        """
        if percentage < 0.01:
            return f"{percentage:.3f}%"  # Display 3 decimal places
        elif percentage < 0.1:
            return f"{percentage:.2f}%"  # Display 2 decimal places
        else:
            return f"{percentage:.1f}%"  # Display 1 decimal place
        
    
    def _summary(self, data_dict: dict):
        """
        Summary of data_dict
        Args:
            data_dict (dict): data_dict
        """
        total_count = sum(
            cert_info["count"] for market in data_dict.values()
            for cert_info in market.values()
        )

        logger.info("LLM data statistics summary:")
        logger.info(f"Total records: {total_count:,}")

        logger.info("\nDistribution of records by language:")
        for market, certifications in data_dict.items():
            market_count = sum(cert_info["count"] for cert_info in certifications.values())
            logger.info(f"{market}: {market_count:,} ({market_count/total_count*100:.1f}%)")

        logger.info("\nDistribution of records by certification type:")
        cert_counts = {}
        for market, certifications in data_dict.items():
            for cert_name, cert_info in certifications.items():
                cert_counts[cert_name] = cert_info["count"]

        for cert_name, count in sorted(cert_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{cert_name}: {count:,} ({count/total_count*100:.1f}%)")
        
    def plot(self,
             data_dict: dict,
             savefig: str = 'llm_dataset.pdf',
             **kwargs):
        
        """Create enhanced sunburst, display LLM dataset distribution"""
    
        color_scheme = self._create_hierarchical_color_scheme()
        
        # Build hierarchical data
        labels = []
        parents = []
        values = []
        colors = []
        ids = []
        
        # Initialize dictionary to store cumulative values for each level
        level_values = {}
        
        # Calculate total count
        total_count = sum(
            cert_info["count"] for market in data_dict.values()
            for cert_info in market.values()
        )
        
        # Set minimum percentage threshold (e.g. 1%)
        min_percentage_threshold = 1.0
        min_count_threshold = (min_percentage_threshold / 100) * total_count
        
        # Iterate through data to build hierarchical structure
        for market, certifications in data_dict.items():
            for cert_name, cert_info in certifications.items():
                count = cert_info["count"]
                description = cert_info["description"]
                
                # Calculate percentage
                percentage = (count / total_count) * 100
                
                # If percentage is less than threshold, use minimum threshold value
                display_count = max(count, min_count_threshold)
                display_percentage = max(percentage, min_percentage_threshold)
                
                # Level 1: Language type
                if market not in level_values:
                    level_values[market] = 0
                level_values[market] += display_count  # Use display value
                
                if market not in labels:
                    labels.append(market)  # Add first, then update
                    parents.append('')
                    values.append(0)  # Set to 0 first, then update
                    ids.append(market)
                    colors.append(color_scheme[market]['base'])
                
                # Level 2: Certification information (outermost level)
                # Calculate percentage based on display_count (for pie chart size)
                # But display real percentage in labels
                real_percentage = (cert_info["count"] / total_count) * 100
                cert_label = f"{cert_name}<br>{cert_info['description']}<br>{self._format_percentage(real_percentage)}"
                cert_id = f"{market}>{cert_label}"
                labels.append(cert_label)
                parents.append(market)
                values.append(display_count)  # Use display value (for pie chart size)
                ids.append(cert_id)
                colors.append(color_scheme[market]['certifications'][cert_name])
        
        # Update values and labels for intermediate levels
        for i, id_val in enumerate(ids):
            if id_val in level_values:
                values[i] = level_values[id_val]
                # Add percentage information for intermediate levels
                if id_val in ['EN', 'CN']:  # Language level
                    # Calculate real market total value
                    real_market_total = sum(
                        cert_info["count"] for market_name, certs in data_dict.items()
                        for cert_info in certs.values()
                        if market_name == id_val
                    )
                    percentage = (real_market_total / total_count) * 100
                    labels[i] = f"{labels[i]}<br>{self._format_percentage(percentage)}"
        
        # Create figure
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            ids=ids,
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>' +
                        'Number: %{value:,}<br>' +
                        'Percentage of parent: %{percentParent}<br>' +
                        'Percentage of total: %{percentRoot}<extra></extra>',
            marker=dict(
                colors=colors,
                line=dict(color="#FFFFFF", width=1)
            ),
            textfont=dict(size=22),
            insidetextorientation='radial',  # Change to radial, make outermost text vertical to tangent
            textinfo='label',  # Display label text
            rotation=90
        ))
        
        # Set layout
        fig.update_layout(
            title={
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial Black', 'color': '#2c3e50'}
            },
            font=dict(size=10, family="Arial"),
            margin=dict(t=100, l=40, r=40, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white',
            width=1000,
            height=800
        )
        
        # Save as high-quality image
        fig.write_image(
            savefig,
            width=1200,
            height=900,
            scale=2,  # Increase resolution
            engine="kaleido"
        )
        
        self._summary(data_dict)
        
        return fig
    
    
if __name__ == "__main__":
    data_dict = {
        "US Stock": {
            "Alpaca": {
                "DJ30": {
                    "Price": {
                        "1d": (68005, "1995-05-01", "2025-05-01"),
                        "1m": (30278827, "2015-05-01", "2025-05-01")
                    },
                    "News": {
                        "1d": (93346, "2015-05-01", "2025-05-01")
                    }
                },
                "SP500": {
                    "Price": {
                        "1d": (951523, "1995-05-01", "2025-05-01"),
                        "1m": (362745068, "2015-05-01", "2025-05-01")
                    },
                    "News": {
                        "1d": (378689, "2015-05-01", "2025-05-01")
                    }
                }
            },
            "FMP": {
                "DJ30": {
                    "Price": {
                        "1d": (212420, "1995-05-01", "2025-05-01"),
                        "1m": (28270266, "2015-05-01", "2025-05-01")
                    },
                    "News": {
                        "1d": (125300, "2015-05-01", "2025-05-01")
                    }
                },
                "SP500": {
                    "Price": {
                        "1d": (2928155, "1995-05-01", "2025-05-01"),
                        "1m": (365404838, "2015-05-01", "2025-05-01")
                    },
                    "News": {
                        "1d": (509890, "2015-05-01", "2025-05-01")
                    }
                }
            }
        },
        "CN Stock": {
            "TuShare": {
                "SSE50": {
                    "Price": {
                        "1d": (214740, "1995-05-01", "2025-05-01")
                    }
                },
                "HS300": {
                    "Price": {
                        "1d": (982272, "1995-05-01", "2025-05-01")
                    }
                }
            },
            "AkShare": {
                "SSE50": {
                    "Price": {
                        "1d": (218546, "1995-05-01", "2025-05-01")
                    }
                },
                "HS300": {
                    "Price": {
                        "1d": (1010735, "1995-05-01", "2025-05-01")
                    }
                }
            },
            "FMP": {
                "SSE50": {
                    "Price": {
                        "1d": (201323, "1995-05-01", "2025-05-01"),
                        "1m": (3842406, "2015-05-01", "2025-05-01")
                    }
                },
                "HS300": {
                    "Price": {
                        "1d": (944337, "1995-05-01", "2025-05-01"),
                        "1m": (20181932, "2015-05-01", "2025-05-01")
                    }
                }
            }
        }
    }
    
    plot = PlotDownloadData()
    plot(data_dict=data_dict, savefig='stock_dataset.pdf')
    
    data_dict = {
        'EN': {
            'ACCA': {
                'count': 2035,
                'description': 'Association of Chartered Certified Accountants'
            },
            'FINANCE': {
                'count': 19385,
                'description': 'MMLU-Finance'
            },
            'CFA': {
                'count': 3136,
                'description': 'Chartered Financial Analyst'
            },
            'CRA': {
                'count': 11769,
                'description': 'Credit Risk Assessment'
            },
            'FRM': {
                'count': 1244,
                'description': 'Financial Risk Manager'
            },
            'ESG': {
                'count': 300,
                'description': 'Environmental, Social, and Governance'
            },
            'TR': {
                'count': 1544,
                'description': 'Travel Insurance'
            },
            'SM': {
                'count': 6824,
                'description': 'Stock Movement'
            }
        },
        'CN': {
            'AMAC': {
                'count': 3027,
                'description': 'Asset Management Association of China'
            },
            'AFP': {
                'count': 1312,
                'description': 'Associate Financial Planner'
            },
            'CCBP_I': {
                'count': 3818,
                'description': 'Certification of China Banking Professional (Intermediate)'
            },
            'CCBP_P': {
                'count': 4069,
                'description': 'Certification of China Banking Professional (Preliminary)'
            },
            'CFQ': {
                'count': 2095,
                'description': 'Certificate of Futures Qualification'
            },
            'CSP': {
                'count': 1377,
                'description': 'Certification of Securities Professional'
            },
            'CCA': {
                'count': 116,
                'description': 'Certified China Actuary'
            },
            'CPA': {
                'count': 4100,
                'description': 'Certified Public Accountant'
            },
            'CIC': {
                'count': 627,
                'description': 'China Insurance Certification'
            },
            'CCDE': {
                'count': 547,
                'description': 'Counterfeit Currency Detection Exam'
            },
            'GTQC': {
                'count': 765,
                'description': 'Gold Trading Qualification Certificate'
            },
            'IEPQ': {
                'count': 5588,
                'description': 'Intermediate Economics Professional Qualification'
            },
            'JEPQ': {
                'count': 3687,
                'description': 'Junior Economics Professional Qualification'
            },
            'JAPQ': {
                'count': 2556,
                'description': 'Junior Actuary Professional Qualification'
            },
            'SSE': {
                'count': 1086,
                'description': 'Securities Special Examination'
            },
            'EXAM': {
                'count': 1056,
                'description': 'Examination'
            }
        }
    }
    
    plot = PlotLLMData()
    plot(data_dict=data_dict, savefig='llm_dataset.pdf')
