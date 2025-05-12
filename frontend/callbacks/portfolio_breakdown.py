import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback

logger = logging.getLogger(__name__)

def register_callbacks(app, controller):
    """Register callbacks for portfolio breakdown visualizations"""
    
    @app.callback(
        Output('breakdown-data-store', 'data'),
        [Input('positions-data-store', 'data')]
    )
    def update_breakdown_data(positions_data):
        """Process position data for breakdown visualizations"""
        if not positions_data:
            logger.warning("No position data available for breakdown visualizations")
            return {
                'sector_data': [],
                'country_data': [],
                'currency_data': []
            }
        
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(positions_data)
            
            # Prepare sector breakdown data
            sector_data = []
            if 'sector' in df.columns and 'market_value' in df.columns:
                # Group by sector and sum market values
                sector_grouped = df.groupby('sector')['market_value'].sum().reset_index()
                # Sort by market value descending
                sector_grouped = sector_grouped.sort_values('market_value', ascending=False)
                # Convert to records
                sector_data = sector_grouped.to_dict('records')
                logger.info(f"Prepared sector breakdown with {len(sector_data)} sectors")
            
            # Prepare country breakdown data
            country_data = []
            if 'country' in df.columns and 'market_value' in df.columns:
                # Group by country and sum market values
                country_grouped = df.groupby('country')['market_value'].sum().reset_index()
                # Sort by market value descending
                country_grouped = country_grouped.sort_values('market_value', ascending=False)
                # Convert to records
                country_data = country_grouped.to_dict('records')
                logger.info(f"Prepared country breakdown with {len(country_data)} countries")
            
            # Prepare currency breakdown data
            currency_data = []
            if 'purchase_currency' in df.columns and 'market_value' in df.columns:
                # Group by currency and sum market values
                currency_grouped = df.groupby('purchase_currency')['market_value'].sum().reset_index()
                # Sort by market value descending
                currency_grouped = currency_grouped.sort_values('market_value', ascending=False)
                # Convert to records
                currency_data = currency_grouped.to_dict('records')
                logger.info(f"Prepared currency breakdown with {len(currency_data)} currencies")
            
            return {
                'sector_data': sector_data,
                'country_data': country_data,
                'currency_data': currency_data
            }
        except Exception as e:
            logger.error(f"Error preparing breakdown data: {str(e)}")
            return {
                'sector_data': [],
                'country_data': [],
                'currency_data': []
            }
    
    @app.callback(
        Output('sector-breakdown-chart', 'figure'),
        [Input('breakdown-data-store', 'data')]
    )
    def update_sector_chart(breakdown_data):
        """Update the sector breakdown pie chart"""
        if not breakdown_data or not breakdown_data.get('sector_data'):
            # Return empty figure
            return go.Figure().update_layout(
                title="No Sector Data Available",
                template="plotly_white"
            )
        
        try:
            # Create DataFrame from sector data
            sector_data = pd.DataFrame(breakdown_data['sector_data'])
            
            # Create pie chart
            fig = px.pie(
                sector_data, 
                values='market_value', 
                names='sector',
                title='Portfolio Allocation by Sector',
                hole=0.4,  # Donut chart
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            # Update layout
            fig.update_layout(
                legend_title="Sectors",
                template="plotly_white",
                margin=dict(t=50, b=20, l=20, r=20)
            )
            
            # Add percentage labels
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hoverinfo='label+percent+value',
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            
            logger.info("Sector breakdown chart created successfully")
            return fig
        except Exception as e:
            logger.error(f"Error creating sector breakdown chart: {str(e)}")
            return go.Figure().update_layout(
                title="Error Creating Sector Chart",
                template="plotly_white"
            )
    
    @app.callback(
        Output('country-breakdown-chart', 'figure'),
        [Input('breakdown-data-store', 'data')]
    )
    def update_country_chart(breakdown_data):
        """Update the country breakdown pie chart"""
        if not breakdown_data or not breakdown_data.get('country_data'):
            # Return empty figure
            return go.Figure().update_layout(
                title="No Country Data Available",
                template="plotly_white"
            )
        
        try:
            # Create DataFrame from country data
            country_data = pd.DataFrame(breakdown_data['country_data'])
            
            # Create pie chart
            fig = px.pie(
                country_data, 
                values='market_value', 
                names='country',
                title='Geographic Exposure by Country',
                hole=0.4,  # Donut chart
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # Update layout
            fig.update_layout(
                legend_title="Countries",
                template="plotly_white",
                margin=dict(t=50, b=20, l=20, r=20)
            )
            
            # Add percentage labels
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hoverinfo='label+percent+value',
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            
            logger.info("Country breakdown chart created successfully")
            return fig
        except Exception as e:
            logger.error(f"Error creating country breakdown chart: {str(e)}")
            return go.Figure().update_layout(
                title="Error Creating Country Chart",
                template="plotly_white"
            )
    
    @app.callback(
        Output('currency-breakdown-chart', 'figure'),
        [Input('breakdown-data-store', 'data')]
    )
    def update_currency_chart(breakdown_data):
        """Update the currency breakdown chart"""
        if not breakdown_data or not breakdown_data.get('currency_data'):
            # Return empty figure
            return go.Figure().update_layout(
                title="No Currency Data Available",
                template="plotly_white"
            )
        
        try:
            # Create DataFrame from currency data
            currency_data = pd.DataFrame(breakdown_data['currency_data'])
            
            # Create bar chart instead of pie for currency
            fig = px.bar(
                currency_data,
                x='purchase_currency',
                y='market_value',
                title='Portfolio Exposure by Currency',
                color='purchase_currency',
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Currency",
                yaxis_title="Market Value (€)",
                legend_title="Currencies",
                template="plotly_white",
                margin=dict(t=50, b=20, l=20, r=20)
            )
            
            # Add value labels on top of bars
            fig.update_traces(
                texttemplate='%{y:.2f}',
                textposition='outside'
            )
            
            logger.info("Currency breakdown chart created successfully")
            return fig
        except Exception as e:
            logger.error(f"Error creating currency breakdown chart: {str(e)}")
            return go.Figure().update_layout(
                title="Error Creating Currency Chart",
                template="plotly_white"
            ) 