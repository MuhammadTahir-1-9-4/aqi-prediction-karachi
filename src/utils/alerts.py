"""
AQI Alert System
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from datetime import datetime
from dotenv import load_dotenv
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQIAlertSystem:
    """AQI Alert System for health warnings"""

    def __init__(self):
        self.email_enabled = bool(os.getenv("ALERT_EMAIL"))
        self.slack_enabled = bool(os.getenv("SLACK_WEBHOOK"))
        self.telegram_enabled = bool(os.getenv("TELEGRAM_BOT_TOKEN"))

        # Alert thresholds
        self.thresholds = {
            "moderate": 2.0,
            "unhealthy_sensitive": 3.0,
            "unhealthy": 4.0,
            "hazardous": 5.0
        }

    def check_alert_condition(self, current_aqi, previous_aqi=None):
        """
        Check if alert should be triggered
        Returns: (alert_level, should_alert)
        """
        if current_aqi >= self.thresholds["hazardous"]:
            return "hazardous", True
        elif current_aqi >= self.thresholds["unhealthy"]:
            return "unhealthy", True
        elif current_aqi >= self.thresholds["unhealthy_sensitive"]:
            return "unhealthy_sensitive", True
        elif current_aqi >= self.thresholds["moderate"]:
            return "moderate", True

        return "good", False

    def get_alert_message(self, alert_level, aqi_value, location="Karachi"):
        """Generate alert message based on level"""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        messages = {
            "hazardous": f"""
🚨 HAZARDOUS AIR QUALITY ALERT 🚨

Location: {location}
AQI: {aqi_value:.2f}
Time: {timestamp}

⚠️ HEALTH IMPACTS:
• Serious respiratory effects
• Aggravation of heart/lung diseases
• Risk of premature death

🛡️ RECOMMENDED ACTIONS:
• Avoid all outdoor activities
• Keep windows and doors closed
• Use air purifiers
• Sensitive groups stay indoors
• Follow local health advisories

This is a health emergency. Take immediate precautions.
            """,

            "unhealthy": f"""
🔴 UNHEALTHY AIR QUALITY ALERT

Location: {location}
AQI: {aqi_value:.2f}
Time: {timestamp}

⚠️ HEALTH IMPACTS:
• Breathing discomfort for everyone
• Aggravation of respiratory conditions
• Increased risk for sensitive groups

🛡️ RECOMMENDED ACTIONS:
• Avoid prolonged outdoor exertion
• People with respiratory issues should stay indoors
• Close windows during peak pollution hours
• Consider rescheduling outdoor activities

Monitor air quality and limit exposure.
            """,

            "unhealthy_sensitive": f"""
🟠 UNHEALTHY FOR SENSITIVE GROUPS ALERT

Location: {location}
AQI: {aqi_value:.2f}
Time: {timestamp}

⚠️ HEALTH IMPACTS:
• May affect sensitive individuals
• Children and elderly at higher risk
• People with respiratory diseases affected

🛡️ RECOMMENDED ACTIONS:
• Sensitive groups should limit outdoor activities
• Reduce prolonged exertion outdoors
• Monitor symptoms closely
• Consider indoor alternatives

Generally safe for most healthy adults.
            """,

            "moderate": f"""
🟡 MODERATE AIR QUALITY ALERT

Location: {location}
AQI: {aqi_value:.2f}
Time: {timestamp}

ℹ️ HEALTH IMPACTS:
• Acceptable air quality for most people
• May affect very sensitive individuals

🛡️ RECOMMENDED ACTIONS:
• Unusually sensitive people should consider limiting prolonged outdoor exertion
• Generally safe for outdoor activities

No major restrictions needed.
            """
        }

        return messages.get(alert_level, "Unknown alert level")

    def send_email_alert(self, subject, message):
        """Send email alert"""
        if not self.email_enabled:
            return False

        try:
            sender_email = os.getenv("ALERT_EMAIL")
            sender_password = os.getenv("ALERT_EMAIL_PASSWORD")
            recipient_emails = os.getenv("ALERT_RECIPIENTS", "").split(",")

            if not all([sender_email, sender_password, recipient_emails]):
                logger.warning("Email alert not configured properly")
                return False

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ", ".join(recipient_emails)
            msg['Subject'] = subject

            msg.attach(MIMEText(message, 'plain'))

            # Send email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, recipient_emails, text)
            server.quit()

            logger.info(f"Email alert sent to {len(recipient_emails)} recipients")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False

    def send_slack_alert(self, message):
        if not self.slack_enabled:
            return False

        try:
            webhook_url = os.getenv("SLACK_WEBHOOK")
            payload = {"text": message}

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("Slack alert sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False

    def send_telegram_alert(self, message):
        if not self.telegram_enabled:
            return False

        try:
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_ids = os.getenv("TELEGRAM_CHAT_IDS", "").split(",")

            if not chat_ids:
                logger.warning("Telegram alert not configured properly")
                return False

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

            success_count = 0
            for chat_id in chat_ids:
                payload = {
                    "chat_id": chat_id.strip(),
                    "text": message,
                    "parse_mode": "HTML"
                }

                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
                success_count += 1

            logger.info(f"Telegram alert sent to {success_count} chats")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {str(e)}")
            return False

    def send_alert(self, alert_level, aqi_value, location="Karachi"):
        """Send alert through all configured channels"""
        message = self.get_alert_message(alert_level, aqi_value, location)
        subject = f"AQI Alert: {alert_level.upper()} - {location}"

        success_count = 0

        if self.send_email_alert(subject, message):
            success_count += 1

        if self.send_slack_alert(message):
            success_count += 1

        if self.send_telegram_alert(message):
            success_count += 1

        if success_count > 0:
            logger.info(f"Alert sent successfully through {success_count} channels")
            return True
        else:
            logger.warning("Failed to send alert through any channel")
            return False

def check_and_alert(current_aqi, location="Karachi"):
    alert_system = AQIAlertSystem()
    alert_level, should_alert = alert_system.check_alert_condition(current_aqi)

    if should_alert:
        return alert_system.send_alert(alert_level, current_aqi, location)

    return False

if __name__ == "__main__":
    print("Testing AQI Alert System...")

    test_aqis = [1.5, 2.5, 3.5, 4.5]

    for aqi in test_aqis:
        alert_system = AQIAlertSystem()
        level, should_alert = alert_system.check_alert_condition(aqi)

        print(f"AQI {aqi}: Level={level}, Alert={should_alert}")

        if should_alert:
            message = alert_system.get_alert_message(level, aqi)
            print(f"Alert message preview: {message[:100]}...")
            print("-" * 50)