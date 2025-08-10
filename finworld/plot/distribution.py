import json
import os.path
from typing import List, Dict
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from glob import glob
import numpy as np
from finworld.utils import assemble_project_path
from finworld.log import logger
from finworld.registry import PLOT

@PLOT.register_module(force=True)
class PlotAssetDistribution():
    def __init__(self):
        super(PlotAssetDistribution, self).__init__()
        self.x_column = "x"
        self.y_column = "y"
        self.z_column = "z"
        
    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)
    
    def _prepare_data(self, data_frames: Dict[str, pd.DataFrame], dimensions: int = 2) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for plotting with specified dimensions
        Args:
            data_frames: Dictionary of stock dataframes
            dimensions: Number of dimensions (2 or 3)
            
        Returns:
            Dict[str, pd.DataFrame]: Prepared dataframes with TSNE coordinates
        """
        data_frames_dict = {}
        for symbol, data_frame in data_frames.items():
            data_frame = data_frame.set_index("timestamp")
            
            # Apply TSNE with specified dimensions
            tsne = TSNE(n_components=dimensions, random_state=1024)
            data_frame_tsne = tsne.fit_transform(data_frame)
            
            if dimensions == 2:
                data_tsne_dict = {
                    "symbol": symbol,
                    self.x_column: data_frame_tsne[:, 0],
                    self.y_column: data_frame_tsne[:, 1],
                }
            elif dimensions == 3:
                data_tsne_dict = {
                    "symbol": symbol,
                    self.x_column: data_frame_tsne[:, 0],
                    self.y_column: data_frame_tsne[:, 1],
                    self.z_column: data_frame_tsne[:, 2],
                }
            else:
                raise ValueError("Dimensions must be 2 or 3")
            
            df_tsne = pd.DataFrame(data_tsne_dict, index=range(len(data_frame)))
            data_frames_dict[symbol] = df_tsne
            
        return data_frames_dict

    def _create_color_scheme(self, symbols: List[str]) -> Dict[str, str]:
        """
        Create a high-contrast color scheme by cyclically selecting from darker yellow, green, and blue color families.

        Args:
            symbols (List[str]): A list of stock or dataset symbols.

        Returns:
            Dict[str, str]: A dictionary mapping each symbol to a unique color.
        """

        # Define darker yellow color family
        yellow_colors = [
            "#FFB74D", "#FFA726", "#FF9800", "#FB8C00",
            "#F57C00", "#EF6C00", "#E65100"
        ]

        # Define darker green color family
        green_colors = [
            "#81C784", "#66BB6A", "#4CAF50", "#43A047",
            "#388E3C", "#2E7D32", "#1B5E20"
        ]

        # Define darker blue color family
        blue_colors = [
            "#64B5F6", "#42A5F5", "#2196F3", "#1E88E5",
            "#1976D2", "#1565C0", "#0D47A1"
        ]

        # Group colors by family for round-robin assignment
        color_families = [yellow_colors, green_colors, blue_colors]

        color_scheme = {}
        family_index = 0  # Tracks which color family to use
        color_indices = [0, 0, 0]  # Individual index for each color family

        for symbol in symbols:
            # Select current color family
            family = color_families[family_index]
            # Select color from the current family
            color = family[color_indices[family_index] % len(family)]
            # Assign color to the symbol
            color_scheme[symbol] = color
            # Update the index for current family
            color_indices[family_index] += 1
            # Rotate to next family
            family_index = (family_index + 1) % len(color_families)

        return color_scheme

        
    def plot(self, 
             data_frames: Dict[str, pd.DataFrame], 
             savefig: str = 'distribution.pdf',
             width_px: int = 1200,
             height_px: int = 800,
             title: str = 'Asset Distribution Plot',
             plot_type: str = 'scatter',  # 'scatter', 'bubble', 'line'
             dimensions: int = 2,  # 2 or 3
             marker_size: int = 8,
             opacity: float = 0.7,
             scene_margin: float = 0.02,  # 3D scene margin control - minimized default
             **kwargs):
        """
        Create 2D or 3D distribution plot for multiple assets
        
        Args:
            data_frames: Dictionary of stock dataframes with keys as symbol names
            savefig: Output file path
            width_px: Image width in pixels
            height_px: Image height in pixels
            title: Plot title
            plot_type: Type of plot ('scatter', 'bubble', 'line')
            dimensions: Number of dimensions (2 or 3)
            marker_size: Size of markers
            opacity: Opacity of markers
            scene_margin: 3D scene margin (0.0-0.5, smaller = less margin)
            **kwargs: Additional arguments
        """
        
        # Validate dimensions
        if dimensions not in [2, 3]:
            raise ValueError("Dimensions must be 2 or 3")
        
        # Prepare data
        prepared_data = self._prepare_data(data_frames, dimensions)
        
        if not prepared_data:
            logger.error("No valid data found for plotting")
            return None
        
        # Create color scheme
        color_scheme = self._create_color_scheme(list(prepared_data.keys()))
        
        # Create figure
        if dimensions == 2:
            fig = go.Figure()
        else:  # 3D
            fig = go.Figure()
        
        # Add traces for each stock
        for symbol, df in prepared_data.items():
            if dimensions == 2:
                # 2D plotting
                if plot_type == 'scatter':
                    fig.add_trace(go.Scatter(
                        x=df[self.x_column],
                        y=df[self.y_column],
                        mode='markers',
                        name=symbol,
                        marker=dict(
                            color=color_scheme[symbol],
                            size=marker_size,
                            opacity=opacity,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>{symbol}</b><br>' +
                                     f'{self.x_column}: %{{x}}<br>' +
                                     f'{self.y_column}: %{{y}}<br>' +
                                     '<extra></extra>'
                    ))
                elif plot_type == 'bubble':
                    size_column = kwargs.get('size_column', None)
                    if size_column and size_column in df.columns:
                        sizes = df[size_column] / df[size_column].max() * 20
                    else:
                        sizes = [marker_size] * len(df)
                    
                    fig.add_trace(go.Scatter(
                        x=df[self.x_column],
                        y=df[self.y_column],
                        mode='markers',
                        name=symbol,
                        marker=dict(
                            color=color_scheme[symbol],
                            size=sizes,
                            opacity=opacity,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>{symbol}</b><br>' +
                                     f'{self.x_column}: %{{x}}<br>' +
                                     f'{self.y_column}: %{{y}}<br>' +
                                     '<extra></extra>'
                    ))
                elif plot_type == 'line':
                    fig.add_trace(go.Scatter(
                        x=df[self.x_column],
                        y=df[self.y_column],
                        mode='lines+markers',
                        name=symbol,
                        line=dict(color=color_scheme[symbol], width=2),
                        marker=dict(
                            color=color_scheme[symbol],
                            size=marker_size,
                            opacity=opacity
                        ),
                        hovertemplate=f'<b>{symbol}</b><br>' +
                                     f'{self.x_column}: %{{x}}<br>' +
                                     f'{self.y_column}: %{{y}}<br>' +
                                     '<extra></extra>'
                    ))
            else:
                # 3D plotting
                if plot_type == 'scatter':
                    fig.add_trace(go.Scatter3d(
                        x=df[self.x_column],
                        y=df[self.y_column],
                        z=df[self.z_column],
                        mode='markers',
                        name=symbol,
                        marker=dict(
                            color=color_scheme[symbol],
                            size=marker_size,
                            opacity=opacity,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>{symbol}</b><br>' +
                                     f'{self.x_column}: %{{x}}<br>' +
                                     f'{self.y_column}: %{{y}}<br>' +
                                     f'{self.z_column}: %{{z}}<br>' +
                                     '<extra></extra>'
                    ))
                elif plot_type == 'bubble':
                    size_column = kwargs.get('size_column', None)
                    if size_column and size_column in df.columns:
                        sizes = df[size_column] / df[size_column].max() * 20
                    else:
                        sizes = [marker_size] * len(df)
                    
                    fig.add_trace(go.Scatter3d(
                        x=df[self.x_column],
                        y=df[self.y_column],
                        z=df[self.z_column],
                        mode='markers',
                        name=symbol,
                        marker=dict(
                            color=color_scheme[symbol],
                            size=sizes,
                            opacity=opacity,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>{symbol}</b><br>' +
                                     f'{self.x_column}: %{{x}}<br>' +
                                     f'{self.y_column}: %{{y}}<br>' +
                                     f'{self.z_column}: %{{z}}<br>' +
                                     '<extra></extra>'
                    ))
                elif plot_type == 'line':
                    fig.add_trace(go.Scatter3d(
                        x=df[self.x_column],
                        y=df[self.y_column],
                        z=df[self.z_column],
                        mode='lines+markers',
                        name=symbol,
                        line=dict(color=color_scheme[symbol], width=2),
                        marker=dict(
                            color=color_scheme[symbol],
                            size=marker_size,
                            opacity=opacity
                        ),
                        hovertemplate=f'<b>{symbol}</b><br>' +
                                     f'{self.x_column}: %{{x}}<br>' +
                                     f'{self.y_column}: %{{y}}<br>' +
                                     f'{self.z_column}: %{{z}}<br>' +
                                     '<extra></extra>'
                    ))
        
        # Update layout
        layout_kwargs = {
            'title': {
                'text': title,
                'x': 0.5,
                'y': 0.95,  # Move title down
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'family': 'Arial Black', 'color': '#2c3e50'}
            },
            'template': 'simple_white',
            'font': dict(size=12, family="Arial"),
            'margin': dict(t=100, l=60, r=40, b=60),
            'paper_bgcolor': 'white',
            'plot_bgcolor': 'white',
            'width': width_px,
            'height': height_px,
            'showlegend': True,
            'legend': dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            )
        }
        
        if dimensions == 2:
            layout_kwargs.update({
                'xaxis_title': self.x_column.title(),
                'yaxis_title': self.y_column.title(),
            })
        else:
            # Compute domain for 3D scene - minimize blank space
            domain_start = scene_margin
            domain_end = 1.0 - scene_margin
            
            layout_kwargs.update({
                'scene': dict(
                    xaxis_title=self.x_column.title(),
                    yaxis_title=self.y_column.title(),
                    zaxis_title=self.z_column.title(),
                    camera=dict(
                        eye=dict(x=1.0, y=1.0, z=1.0)  # Closer camera to make data appear larger
                    ),
                    # Minimize 3D scene margin to maximize data area
                    domain=dict(x=[domain_start, domain_end], y=[domain_start, domain_end]),
                    aspectmode='data',  # Adjust aspect ratio based on data
                    aspectratio=dict(x=1, y=1, z=1),
                    # Reduce axis line margin
                    xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=1),
                    yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=1),
                    zaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=1)
                ),
                # Minimize overall margin to maximize plot area
                'margin': dict(t=20, l=5, r=5, b=5),
                # Adjust 3D scene position and size on canvas
                'width': width_px,
                'height': height_px,
            })
        
        fig.update_layout(**layout_kwargs)
        
        # Save as high-quality image
        fig.write_image(
            savefig,
            width=width_px,
            height=height_px,
            scale=2,  # Increase resolution
            engine="kaleido"
        )
        
        logger.info(f"{dimensions}D distribution plot saved to: {savefig}")
        logger.info(f"Plotted {len(prepared_data)} assets in {dimensions}D")
        
        return fig
    

@PLOT.register_module(force=True)
class PlotPoolDistribution():
    def __init__(self):
        super(PlotPoolDistribution, self).__init__()
        self.x_column = "x"
        self.y_column = "y"
        self.z_column = "z"
        
    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)
    
    def _prepare_data(self, data_frames: Dict[str, pd.DataFrame], dimensions: int = 2) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for plotting with specified dimensions
        Args:
            data_frames: Dictionary of stock dataframes
            dimensions: Number of dimensions (2 or 3)
            
        Returns:
            Dict[str, pd.DataFrame]: Prepared dataframes with TSNE coordinates
        """
        data_frames_dict = {}
        for pool_name, pool_data_frames in data_frames.items():
            
            if dimensions == 2:
                pool_data_frame_dict = {
                "symbol": [],
                    self.x_column: [],
                    self.y_column: []
                }
            elif dimensions == 3:
                pool_data_frame_dict = {
                    "symbol": [],
                    self.x_column: [],
                    self.y_column: [],
                    self.z_column: []
                }
            else:
                raise ValueError("Dimensions must be 2 or 3")
            
            for symbol, symbol_data_frame in pool_data_frames.items():
                symbol_data_frame = symbol_data_frame.set_index("timestamp")
                tsne = TSNE(n_components=dimensions, random_state=1024)
                symbol_data_frame_tsne = tsne.fit_transform(symbol_data_frame)
                
                # Average the TSNE coordinates of the symbol
                symbol_data_mean = symbol_data_frame_tsne.mean(axis=0)
                
                pool_data_frame_dict["symbol"].append(symbol)
                if dimensions == 2:
                    pool_data_frame_dict[self.x_column].append(symbol_data_mean[0])
                    pool_data_frame_dict[self.y_column].append(symbol_data_mean[1])
                elif dimensions == 3:
                    pool_data_frame_dict[self.x_column].append(symbol_data_mean[0])
                    pool_data_frame_dict[self.y_column].append(symbol_data_mean[1])
                    pool_data_frame_dict[self.z_column].append(symbol_data_mean[2])
                else:
                    raise ValueError("Dimensions must be 2 or 3")
                
            data_frames_dict[pool_name] = pd.DataFrame(pool_data_frame_dict, index=range(len(pool_data_frames)))
            
        return data_frames_dict
    
    
    def _create_color_scheme(self, pools: List[str]) -> Dict[str, str]:
        """
        Create a high-contrast color scheme by cyclically selecting from darker yellow, green, and blue color families.

        Args:
            pools (List[str]): A list of pool names.

        Returns:
            Dict[str, str]: A dictionary mapping each pool to a unique color.
        """

        # Define darker yellow color family
        yellow_colors = [
            "#FFB74D", "#FFA726", "#FF9800", "#FB8C00",
            "#F57C00", "#EF6C00", "#E65100"
        ]

        # Define darker green color family
        green_colors = [
            "#81C784", "#66BB6A", "#4CAF50", "#43A047",
            "#388E3C", "#2E7D32", "#1B5E20"
        ]

        # Define darker blue color family
        blue_colors = [
            "#64B5F6", "#42A5F5", "#2196F3", "#1E88E5",
            "#1976D2", "#1565C0", "#0D47A1"
        ]

        # Group colors by family for round-robin assignment
        color_families = [yellow_colors, green_colors, blue_colors]

        color_scheme = {}
        family_index = 0  # Tracks which color family to use
        color_indices = [0, 0, 0]  # Individual index for each color family

        for pool in pools:
            # Select current color family
            family = color_families[family_index]
            # Select color from the current family
            color = family[color_indices[family_index] % len(family)]
            # Assign color to the symbol
            color_scheme[pool] = color
            # Update the index for current family
            color_indices[family_index] += 1
            # Rotate to next family
            family_index = (family_index + 1) % len(color_families)

        return color_scheme

    
    def plot(self, 
             data_frames: Dict[str, pd.DataFrame], 
             savefig: str = 'distribution.pdf',
             width_px: int = 1200,
             height_px: int = 800,
             title: str = 'Asset Distribution Plot',
             plot_type: str = 'scatter',  # 'scatter', 'bubble', 'line'
             dimensions: int = 2,  # 2 or 3
             marker_size: int = 8,
             opacity: float = 0.7,
             scene_margin: float = 0.02,  # 3D scene margin control - minimized default
             **kwargs):
        """
        Create 2D or 3D distribution plot for multiple assets
        
        Args:
            data_frames: Dictionary of stock dataframes with keys as symbol names
            savefig: Output file path
            width_px: Image width in pixels
            height_px: Image height in pixels
            title: Plot title
            plot_type: Type of plot ('scatter', 'bubble', 'line')
            dimensions: Number of dimensions (2 or 3)
            marker_size: Size of markers
            opacity: Opacity of markers
            scene_margin: 3D scene margin (0.0-0.5, smaller = less margin)
            **kwargs: Additional arguments
        """
        
        # Validate dimensions
        if dimensions not in [2, 3]:
            raise ValueError("Dimensions must be 2 or 3")
        
        # Prepare data
        prepared_data = self._prepare_data(data_frames, dimensions)
        
        if not prepared_data:
            logger.error("No valid data found for plotting")
            return None
        
        # Create color scheme
        color_scheme = self._create_color_scheme(list(prepared_data.keys()))
        
        # Create figure
        if dimensions == 2:
            fig = go.Figure()
        else:  # 3D
            fig = go.Figure()
        
        # Add traces for each stock
        for pool_name, pool_data_frames in prepared_data.items():
            if dimensions == 2:
                # 2D plotting
                if plot_type == 'scatter':
                    fig.add_trace(go.Scatter(
                        x=pool_data_frames[self.x_column],
                        y=pool_data_frames[self.y_column],
                        mode='markers',
                        name=pool_name,
                        marker=dict(
                            color=color_scheme[pool_name],
                            size=marker_size,
                            opacity=opacity,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>{pool_name}</b><br>' +
                                     f'{self.x_column}: %{{x}}<br>' +
                                     f'{self.y_column}: %{{y}}<br>' +
                                     '<extra></extra>'
                    ))
                elif plot_type == 'bubble':
                    size_column = kwargs.get('size_column', None)
                    if size_column and size_column in pool_data_frames.columns:
                        sizes = pool_data_frames[size_column] / pool_data_frames[size_column].max() * 20
                    else:
                        sizes = [marker_size] * len(pool_data_frames)
                    
                    fig.add_trace(go.Scatter(
                        x=pool_data_frames[self.x_column],
                        y=pool_data_frames[self.y_column],
                        mode='markers',
                        name=pool_name,
                        marker=dict(
                            color=color_scheme[pool_name],
                            size=sizes,
                            opacity=opacity,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>{pool_name}</b><br>' +
                                     f'{self.x_column}: %{{x}}<br>' +
                                     f'{self.y_column}: %{{y}}<br>' +
                                     '<extra></extra>'
                    ))
                elif plot_type == 'line':
                    fig.add_trace(go.Scatter(
                        x=pool_data_frames[self.x_column],
                        y=pool_data_frames[self.y_column],
                        mode='lines+markers',
                        name=pool_name,
                            line=dict(color=color_scheme[pool_name], width=2),
                        marker=dict(
                            color=color_scheme[pool_name],
                            size=marker_size,
                            opacity=opacity
                        ),
                        hovertemplate=f'<b>{pool_name}</b><br>' +
                                     f'{self.x_column}: %{{x}}<br>' +
                                     f'{self.y_column}: %{{y}}<br>' +
                                     '<extra></extra>'
                    ))
            else:
                # 3D plotting
                if plot_type == 'scatter':
                    fig.add_trace(go.Scatter3d(
                        x=pool_data_frames[self.x_column],
                        y=pool_data_frames[self.y_column],
                        z=pool_data_frames[self.z_column],
                        mode='markers',
                        name=pool_name,
                        marker=dict(
                            color=color_scheme[pool_name],
                            size=marker_size,
                            opacity=opacity,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>{pool_name}</b><br>' +
                                     f'{self.x_column}: %{{x}}<br>' +
                                     f'{self.y_column}: %{{y}}<br>' +
                                     f'{self.z_column}: %{{z}}<br>' +
                                     '<extra></extra>'
                    ))
                elif plot_type == 'bubble':
                    size_column = kwargs.get('size_column', None)
                    if size_column and size_column in pool_data_frames.columns:
                        sizes = pool_data_frames[size_column] / pool_data_frames[size_column].max() * 20
                    else:
                        sizes = [marker_size] * len(pool_data_frames)
                    
                    fig.add_trace(go.Scatter3d(
                        x=pool_data_frames[self.x_column],
                        y=pool_data_frames[self.y_column],
                        z=pool_data_frames[self.z_column],
                        mode='markers',
                        name=pool_name,
                        marker=dict(
                            color=color_scheme[pool_name],
                            size=sizes,
                            opacity=opacity,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>{pool_name}</b><br>' +
                                     f'{self.x_column}: %{{x}}<br>' +
                                     f'{self.y_column}: %{{y}}<br>' +
                                     f'{self.z_column}: %{{z}}<br>' +
                                     '<extra></extra>'
                    ))
                elif plot_type == 'line':
                    fig.add_trace(go.Scatter3d(
                        x=pool_data_frames[self.x_column],
                        y=pool_data_frames[self.y_column],
                        z=pool_data_frames[self.z_column],
                        mode='lines+markers',
                        name=pool_name,
                        line=dict(color=color_scheme[pool_name], width=2),
                        marker=dict(
                            color=color_scheme[pool_name],
                            size=marker_size,
                            opacity=opacity
                        ),
                        hovertemplate=f'<b>{pool_name}</b><br>' +
                                     f'{self.x_column}: %{{x}}<br>' +
                                     f'{self.y_column}: %{{y}}<br>' +
                                     f'{self.z_column}: %{{z}}<br>' +
                                     '<extra></extra>'
                    ))
        
        # Update layout
        layout_kwargs = {
            'title': {
                'text': title,
                'x': 0.5,
                'y': 0.95,  # Move title down
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'family': 'Arial Black', 'color': '#2c3e50'}
            },
            'template': 'simple_white',
            'font': dict(size=12, family="Arial"),
            'margin': dict(t=100, l=60, r=40, b=60),
            'paper_bgcolor': 'white',
            'plot_bgcolor': 'white',
            'width': width_px,
            'height': height_px,
            'showlegend': True,
            'legend': dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            )
        }
        
        if dimensions == 2:
            layout_kwargs.update({
                'xaxis_title': self.x_column.title(),
                'yaxis_title': self.y_column.title(),
            })
        else:
            # Compute domain for 3D scene - minimize blank space
            domain_start = scene_margin
            domain_end = 1.0 - scene_margin
            
            layout_kwargs.update({
                'scene': dict(
                    xaxis_title=self.x_column.title(),
                    yaxis_title=self.y_column.title(),
                    zaxis_title=self.z_column.title(),
                    camera=dict(
                        eye=dict(x=1.0, y=1.0, z=1.0)  # Closer camera to make data appear larger
                    ),
                    # Minimize 3D scene margin to maximize data area
                    domain=dict(x=[domain_start, domain_end], y=[domain_start, domain_end]),
                    aspectmode='data',  # Adjust aspect ratio based on data
                    aspectratio=dict(x=1, y=1, z=1),
                    # Reduce axis line margin
                    xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=1),
                    yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=1),
                    zaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=1)
                ),
                # Minimize overall margin to maximize plot area
                'margin': dict(t=20, l=5, r=5, b=5),
                # Adjust 3D scene position and size on canvas
                'width': width_px,
                'height': height_px,
            })
        
        fig.update_layout(**layout_kwargs)
        
        # Save as high-quality image
        fig.write_image(
            savefig,
            width=width_px,
            height=height_px,
            scale=2,  # Increase resolution
            engine="kaleido"
        )
        
        logger.info(f"{dimensions}D distribution plot saved to: {savefig}")
        logger.info(f"Plotted {len(prepared_data)} assets in {dimensions}D")
        
        return fig
        

    
if __name__ == "__main__":
    
    # symbols = ["AAPL", "TSLA", "META"]
    # data_frames = {}
    # for symbol in symbols:
    #     data_path = assemble_project_path(os.path.join(f"datasets/exp/exp_fmp_price_1day/{symbol}.jsonl"))
    #     df = pd.read_json(data_path, lines=True)
    #     df['volume'] = np.log1p((df['volume']))
    #     df = df.iloc[-800:]
    #     df['timestamp'] = pd.to_datetime(df['timestamp'])
    #     data_frames[symbol] = df
    
    # plot = PlotAssetDistribution()
    
    
    # # Example 1: 2D bubble plot
    # plot(
    #     data_frames=data_frames, 
    #     savefig="asset_distribution_2d_bubble.pdf",
    #     dimensions=2,
    #     plot_type='bubble',
    #     title='2D Asset Distribution Bubble Plot',
    #     width_px=1400,
    #     height_px=1000,
    #     marker_size=12,
    #     opacity=0.8
    # )
    
    # # Example 2: 3D bubble plot
    # plot(
    #     data_frames=data_frames, 
    #     savefig="asset_distribution_3d_bubble.pdf",
    #     dimensions=3,
    #     plot_type='bubble',
    #     title='3D Asset Distribution Bubble Plot',
    #     width_px=1400,
    #     height_px=1000,
    #     marker_size=12,
    #     opacity=0.8,
    #     scene_margin=0.05  # Reduce 3D scene margin, minimize blank space
    # )
    
    pools = ["sse50", "dj30"]

    data_frames = {}
    for pool in pools:
        if pool == "sse50":
            pool_path = assemble_project_path(os.path.join(f"datasets/{pool}/{pool}_akshare_feature_1day"))
        else:
            pool_path = assemble_project_path(os.path.join(f"datasets/{pool}/{pool}_fmp_feature_1day"))
        pool_data_frames = {}
        for file in glob(os.path.join(pool_path, "*.jsonl")):
            symbol = os.path.basename(file).split(".")[0]
            df = pd.read_json(file, lines=True)
            df = df.iloc[-500:]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            pool_data_frames[symbol] = df
        data_frames[pool] = pool_data_frames
    
    plot = PlotPoolDistribution()
    plot(
        data_frames=data_frames,
        savefig="pool_distribution_2d_bubble.pdf",
        dimensions=2,
        plot_type='bubble',
        title='2D Pool Distribution Bubble Plot',
        width_px=1400,
        height_px=1000,
        marker_size=12,
        opacity=0.8
    )
    
    plot(
        data_frames=data_frames,
        savefig="pool_distribution_3d_bubble.pdf",
        dimensions=3,
        plot_type='bubble',
        title='3D Pool Distribution Bubble Plot',
        width_px=1400,
        height_px=1000,
        marker_size=12,
        opacity=0.8
    )