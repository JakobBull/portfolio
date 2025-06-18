from . import (
    portfolio_summary,
    portfolio_performance,
    portfolio_positions,
    add_position,
    tax_settings,
    watchlist,
    portfolio_breakdown,
    data_management,
    single_stock_chart
)

def register_callbacks(app, controller):
    """Register all callbacks for the application."""
    portfolio_summary.register_callbacks(app, controller)
    portfolio_performance.register_callbacks(app, controller)
    portfolio_positions.register_callbacks(app, controller)
    add_position.register_callbacks(app, controller)
    tax_settings.register_callbacks(app, controller)
    watchlist.register_callbacks(app, controller)
    portfolio_breakdown.register_callbacks(app, controller)
    data_management.register_callbacks(app, controller)
    single_stock_chart.register_callbacks(app, controller)