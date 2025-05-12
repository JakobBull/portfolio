from frontend.callbacks import (
    portfolio_summary,
    portfolio_performance,
    portfolio_positions,
    add_position,
    tax_settings,
    general,
    watchlist,
    portfolio_breakdown
)

def register_all_callbacks(app, controller):
    """Register all callbacks for the app"""
    portfolio_summary.register_callbacks(app, controller)
    portfolio_performance.register_callbacks(app, controller)
    portfolio_positions.register_callbacks(app, controller)
    add_position.register_callbacks(app, controller)
    tax_settings.register_callbacks(app, controller)
    general.register_callbacks(app, controller)
    watchlist.register_callbacks(app, controller)
    portfolio_breakdown.register_callbacks(app, controller)