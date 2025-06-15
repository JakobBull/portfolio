import logging
from dash import Input, Output, State, callback, html
from backend.controller import Controller

def register_callbacks(app, controller: Controller):
    """Register callbacks for the tax settings section."""

    @app.callback(
        Output("tax-update-result", "children"),
        Input("update-tax-button", "n_clicks"),
        State("tax-settings", "value"),
        prevent_initial_call=True,
    )
    def update_tax_settings_callback(n_clicks, settings):
        """Updates tax calculator settings and displays the result."""
        try:
            if controller.update_tax_settings(settings or []):
                return html.Div("Tax settings updated successfully.", className="text-success")
            else:
                return html.Div("Failed to update tax settings.", className="text-danger")
        except Exception as e:
            logging.error(f"Error updating tax settings: {e}")
            return html.Div(f"An error occurred: {e}", className="text-danger") 