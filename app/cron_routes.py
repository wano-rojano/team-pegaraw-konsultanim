"""Cron job routes for weekly advisories."""
from flask import jsonify, request
import asyncio
import logging
import os

from app.weekly_advisory_scheduler import run_weekly_advisory_job

logger = logging.getLogger(__name__)


def register_cron_routes(app):
    """Register cron job routes."""
    
    @app.route('/cron/weekly-advisory', methods=['POST'])
    def trigger_weekly_advisory():
        """Endpoint for Render cron to trigger weekly advisories."""

        # Verify cron secret
        cron_secret = request.headers.get('X-Cron-Secret')
        expected_secret = os.getenv('CRON_SECRET')
        
        if expected_secret and cron_secret != expected_secret:
            logger.warning("Unauthorized cron job attempt")
            return jsonify({'error': 'Unauthorized'}), 401
        
        try:
            logger.info("Weekly advisory cron job triggered")
            
            # Run the async advisory job
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(run_weekly_advisory_job())
            finally:
                loop.close()
            
            return jsonify({
                'status': 'success',
                'message': 'Weekly advisories sent successfully'
            }), 200
            
        except Exception as e:
            logger.error(f"Weekly advisory cron job failed: {e}", exc_info=True)
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500