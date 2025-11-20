"""Weekly advisory scheduler for automated farmer notifications."""
import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List

from twilio.rest import Client

from .advisory_agent import AdvisoryAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeeklyAdvisoryScheduler:
    """Generates and sends weekly advisories to farmers."""

    def __init__(self, dry_run: bool = False):
        self.advisory_agent = AdvisoryAgent()
        self.dry_run = dry_run
        
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        
        self.twilio_client = Client(account_sid, auth_token)
        self.from_number = os.getenv('TWILIO_PHONE_NUMBER')
        
        self.farmer_registry = self._load_farmer_registry()

    def _load_farmer_registry(self) -> Dict[str, Dict[str, str]]:
        """Load farmer registry from environment or database.
        
        Format: phone:location:crop,phone:location:crop
        Example: +639123:Laguna:rice,+639456:Nueva Ecija:corn
        """
        registry = {}
        registry_str = os.getenv('FARMER_REGISTRY', '')
        
        if registry_str:
            for entry in registry_str.split(','):
                parts = entry.strip().split(':')
                if len(parts) >= 3:
                    phone = parts[0].strip()
                    location = parts[1].strip()
                    crop = parts[2].strip()
                    registry[phone] = {
                        'location': location,
                        'crop': crop
                    }
        
        logger.info(f"Registry loaded: {len(registry)} farmers")
        return registry

    async def generate_advisory_for_location(
        self, location: str, crop: str
    ) -> str:
        """Generate weekly advisory for a specific location and crop."""
        
        query = (
            f"Generate a weekly farming advisory for {crop} farmers in {location}. "
            f"\n\n"
            f"Include in your advisory: "
            f"1) 7-day weather forecast (use get_weather_forecast tool) "
            f"2) Disease risk assessment based on forecast conditions for {crop} "
            f"   (use retrieve_crop_information to query {crop} pathology PDFs for evidence-based assessment) "
            f"3) Preventive measures {crop} farmers should take this week "
            f"4) What to monitor in {crop} fields "
            f"\n\n"
            f"Important: Keep response concise for SMS (under 1600 characters). "
            f"Use bullet points. Cite evidence from PDFs. Be actionable."
        )
        
        context_id = f"weekly_advisory_{crop}_{location}_{datetime.now().strftime('%Y%m%d')}"
        
        try:
            full_response = ""
            async for item in self.advisory_agent.stream(query, context_id):
                if item.get('is_task_complete') and not item.get('require_user_input'):
                    full_response = item.get('content', '')
                    break
            
            if not full_response:
                logger.warning(f"Empty response for {crop} in {location}, using fallback")
                return self._generate_fallback_advisory(location, crop)
            
            # Format for SMS
            formatted_advisory = (
                f"WEEKLY ADVISORY - {crop.upper()}\n"
                f"{location}\n"
                f"{datetime.now().strftime('%b %d, %Y')}\n"
                f"{'-'*40}\n\n"
                f"{full_response}\n\n"
                f"Reply with questions. - Konsultanim"
            )
            
            # Truncate if too long for SMS (1600 char limit)
            if len(formatted_advisory) > 1600:
                formatted_advisory = formatted_advisory[:1550] + "\n\n- Konsultanim"
            
            return formatted_advisory
            
        except Exception as e:
            logger.error(f"Advisory generation failed for {crop}/{location}: {e}")
            return self._generate_fallback_advisory(location, crop)

    def _generate_fallback_advisory(self, location: str, crop: str = 'rice') -> str:
        """Generate fallback advisory when agent fails."""
        return (
            f"WEEKLY ADVISORY - {crop.upper()}\n"
            f"{location}\n"
            f"{datetime.now().strftime('%b %d, %Y')}\n"
            f"{'-'*40}\n\n"
            f"Unable to generate detailed forecast.\n\n"
            f"Key reminders for {crop} farmers:\n"
            f"• Monitor fields daily for disease\n"
            f"• Check local weather forecasts\n"
            f"• Maintain proper water management\n"
            f"• Contact local extension office\n\n"
            f"Reply with questions. - Konsultanim"
        )

    async def send_weekly_advisories(self) -> Dict[str, int]:
        """Generate and send weekly advisories to all registered farmers."""
        
        logger.info(f"Weekly advisory job started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Group farmers by (location, crop)
        location_crop_to_farmers: Dict[tuple, List[str]] = {}
        for phone, info in self.farmer_registry.items():
            location = info['location']
            crop = info.get('crop', 'rice')
            
            key = (location, crop)
            if key not in location_crop_to_farmers:
                location_crop_to_farmers[key] = []
            location_crop_to_farmers[key].append(phone)
        
        results = {
            'total_locations': len(location_crop_to_farmers),
            'total_farmers': len(self.farmer_registry),
            'successful': 0,
            'failed': 0
        }
        
        # Process each location-crop combination
        for idx, ((location, crop), farmers) in enumerate(location_crop_to_farmers.items(), 1):
            print(f"[{idx}/{len(location_crop_to_farmers)}] {crop.upper()} in {location} ({len(farmers)} farmers)")
            
            # Generate advisory
            advisory = await self.generate_advisory_for_location(location, crop)

            print(f"\n{'='*70}")
            print(f"ADVISORY PREVIEW - {crop.upper()} in {location}")
            print(advisory)
            print(f"{'='*70}")
            
            location_success = 0
            location_failed = 0
            
            # Send to all farmers in this location-crop combo
            for phone, _ in farmers:
                try:
                    if self.dry_run:
                        logger.info(f"  [DRY RUN] Would send to {phone} ({len(advisory)} chars)")
                        location_success += 1
                    else:
                        self.twilio_client.messages.create(
                            body=advisory,
                            from_=self.from_number,
                            to=phone
                        )
                        location_success += 1
                        logger.info(f"  ✓ Sent to {phone}")
                    
                    results['successful'] += 1
                    
                except Exception as e:
                    location_failed += 1
                    results['failed'] += 1
                    logger.error(f"  ✗ Failed to send to {phone}: {e}")
                
                await asyncio.sleep(1)  # Rate limiting
            
            # Store results
            key_str = f"{location}_{crop}"
            results['by_location'][key_str] = {
                'success': location_success,
                'failed': location_failed
            }
        
        # Summary
        success_rate = (results['successful'] / results['total_farmers'] * 100) if results['total_farmers'] > 0 else 0
        logger.info(
            f"Job complete: {results['successful']}/{results['total_farmers']} sent "
            f"({success_rate:.1f}% success)"
        )
        
        if self.dry_run:
            logger.info("DRY RUN - No actual SMS sent")

        return results


async def run_weekly_advisory_job(dry_run: bool = None):
    """Main entry point for the weekly advisory cron job."""
    if dry_run is None:
        dry_run = os.getenv('DRY_RUN', 'false').lower() == 'true'
    scheduler = WeeklyAdvisoryScheduler(dry_run=dry_run)
    results = await scheduler.send_weekly_advisories()
    return results
