# Telegram Bot for Trash Classification and Reward System

## Overview
This bot promotes environmental awareness by allowing users to classify trash, submit photos and videos, and earn points for responsible waste disposal. Administrators review submissions, and a leaderboard tracks top contributors.

## Features
- **Trash Classification**: Classifies trash using a TensorFlow Lite model. Model can analysie class_names as cardboard, glass, metal, paper, plastic, trash.
- **Geolocation Tracking**: Users can send their location with waste images.
- **Submission Approval**: Admins approve or reject user submissions via inline buttons.
- **Point System**: Users earn points for approved submissions, with a leaderboard to track top contributors.
- **Database Integration**: Tracks user points and submissions using SQLite.

## Requirements
- Python check the runtime.txt
- Telegram Bot API Token
- TensorFlow Lite Runtime
- PIL (Pillow)
- SQLite3
- Nest Asyncio
- Python-Telegram-Bot

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Eraly-ml/eco_AI.git
   cd repository
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export TELEGRAM_BOT_TOKEN=your_token_here
   export ADMIN_CHAT_ID=your_admin_chat_id
   ```
4. Run the bot:
   ```bash
   python eco_tg_bot.py
   ```

## Usage

### User Commands
- `/start`: Register yourself in the system.
- `/my`: View your current points.
- `/stat <user_id>`: Check the points of a specific user.
- `/top`: View the leaderboard.
- `/myid`: Retrieve your Telegram user ID.

### Submission Workflow
1. **Send Location**: Share the location of the waste.
2. **Send Photo**: Upload a photo of the waste.
3. **Send Video**: Upload a video showing proper disposal.
4. **Admin Review**: Submission is reviewed by an admin.
5. **Earn Points**: Approved submissions earn points.

### Admin Controls
- Approve or reject submissions directly from the chat interface.
- Review photos and videos attached to submissions.

## File Structure
```
repository/
├── eco_tg_bot.py              # Main bot script
├── model.tflite        # TensorFlow Lite model
├── requirements.txt    # Python dependencies
├── bot_database.db     # SQLite database (auto-generated)
└── README.md           # Project documentation
```

## Technologies Used
- **Telegram Bot API**: For user interaction.
- **TensorFlow Lite**: For lightweight trash classification.
- **SQLite**: To store user data and track points.
- **Pillow**: For image preprocessing.

## How It Works

1. Users interact with the bot via commands and share location, photos, and videos.
2. The bot classifies the photo of trash by using a pre-trained MobileNet TensorFlow Lite model.
3. Submissions are reviewed by an admin and sendet to private chat of admins.
4. Approved submissions earn points and update the leaderboard.
## information

User data is used exclusively for bot functionality and to evaluate user contributions.
## Contributing

Currently, contribution guidelines are informal. Feel free to reach out via Telegram: @eralyf

## License

This project is licensed under the MIT License. See the LICENSE file for details, also we have liberies like python-telegram-bot from telegram

## Contact
For questions or contact the repository owner use telegram @eralyf

## Support the Developer  

If you'd like to support my work, consider subscribing to my Telegram channel:  
🔗 **[EraJustML](https://t.me/erajustml)** 

Your support means a lot!

