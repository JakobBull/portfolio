import logging
from dash import Input, Output, State, callback, html

def register_callbacks(app, controller):
    """Register callbacks for tax settings section"""
    
    @app.callback(
        Output('tax-result-store', 'data'),
        [Input('update-tax-button', 'n_clicks')],
        [State('tax-settings', 'value')],
        prevent_initial_call=True
    )
    def update_tax_settings(n_clicks, settings):
        """Update tax calculator settings and store the result"""
        if not n_clicks:
            return None
            
        try:
            # Parse settings and update controller
            success = controller.update_tax_settings(settings)
            
            if success:
                return {'success': True, 'message': "Tax settings updated successfully"}
            else:
                return {'success': False, 'message': "Failed to update tax settings"}
        except Exception as e:
            logging.error(f"Error updating tax settings: {str(e)}")
            return {'success': False, 'message': f"Error updating tax settings: {str(e)}"}

    @app.callback(
        Output('tax-update-result', 'children'),
        [Input('tax-result-store', 'data')]
    )
    def update_tax_result(data):
        """Update the tax settings result UI with data from the store"""
        if not data:
            return ""
        
        if data['success']:
            return html.Div(data['message'], className="text-success")
        else:
            return html.Div(data['message'], className="text-danger") 