# Privacy Policy — ChatMind

_Last updated: August 2026_

ChatMind (@chatmind_search_bot) is a Telegram bot that indexes photos shared in a chat so they can be found later by description. This page explains what data the bot collects, why, and how you can remove it.

## What the bot stores

When a photo is sent to a chat the bot is in, it stores:

- **The photo itself** — not the image file, only Telegram's own `file_id` reference. The bot never downloads a permanent copy; it asks Telegram to resend the photo using that reference each time it's shown.
- **Text visible in the photo** (via OCR), so it can be found by that text later.
- **A CLIP embedding** — a numeric vector describing the photo's visual content, used for similarity search. This is not human-readable and cannot be reversed back into the image.
- **Who sent it** — the sender's Telegram user ID and display name, and the chat ID and message ID.
- **Timestamp** and **favorite status** (if toggled with ❤️).

The bot does **not** store your messages, contacts, or anything you haven't sent it directly as a photo or a `/find` search query. Search queries themselves are not stored — they're used once, in memory, to run the search, and discarded.

## Who can see this data

Data is scoped per chat — photos indexed in one chat are never searchable from a different chat. Within a chat, anyone in that chat can search (`/find`), favorite (❤️), or view stats (`/stats`) for photos sent there.

## How to delete your data

Tap **🗑️ Delete** on any photo shown in a search result (`/find` or `/favorites`) — only the person who originally sent that photo can delete it. This immediately and permanently removes it from the bot's index; it will no longer show up in searches. The original photo message in the chat itself is untouched (the bot doesn't have permission to delete Telegram messages).

## Third parties

Photo processing (OCR and CLIP embeddings) runs locally on the bot's own server — no third-party AI service ever receives your photos. The bot uses Telegram's Bot API to send/receive messages, which is subject to [Telegram's own Privacy Policy](https://telegram.org/privacy).