import express, { Request, Response } from 'express';
import crypto from 'crypto';
import { parseStringPromise } from 'xml2js';
import dotenv from 'dotenv';

dotenv.config();

const PORT: number = parseInt(process.env.PORT ?? '3000', 10);
const WECHAT_TOKEN: string = process.env.WECHAT_TOKEN ?? '';
const OPENROUTER_API_KEY: string = process.env.OPENROUTER_API_KEY ?? '';

if (!WECHAT_TOKEN) {
  console.error('Missing WECHAT_TOKEN environment variable.');
  process.exit(1);
}

if (!OPENROUTER_API_KEY) {
  console.warn('OPENROUTER_API_KEY is not set; LLM features will be unavailable.');
}

const app = express();

// Parse raw XML bodies sent by the WeChat server
app.use(express.text({ type: 'text/xml' }));

// ---------- Types ----------

interface WeChatTextMessage {
  ToUserName: string;
  FromUserName: string;
  CreateTime: string;
  MsgType: string;
  Content?: string;
  MsgId?: string;
}

interface ParsedWeChatEnvelope {
  xml: WeChatTextMessage;
}

type CommandType = 'vocab' | 'translate' | 'chat';

interface CommandPayload {
  fromUser: string;
  toUser: string;
  argument: string;
}

// ---------- Helpers ----------

/**
 * Verify a WeChat signature by sorting the token, timestamp, and nonce
 * lexicographically, joining them, and SHA1-hashing the result.
 */
function verifyWeChatSignature(
  token: string,
  timestamp: string,
  nonce: string,
  signature: string,
): boolean {
  const joined = [token, timestamp, nonce].sort().join('');
  const hash = crypto.createHash('sha1').update(joined).digest('hex');
  return hash === signature;
}

/**
 * Route a user message to a command type and extract its argument.
 */
function routeCommand(content: string): { type: CommandType; argument: string } {
  if (content.startsWith('#vocab ')) {
    return { type: 'vocab', argument: content.slice('#vocab '.length).trim() };
  }
  if (content.startsWith('#translate ')) {
    return { type: 'translate', argument: content.slice('#translate '.length).trim() };
  }
  return { type: 'chat', argument: content };
}

/**
 * Mock command processor. Returns a static string until the LLM integration
 * is implemented.
 */
async function processCommand(command: CommandType, payload: CommandPayload): Promise<string> {
  switch (command) {
    case 'vocab':
      return `[mock vocab] Definition lookup for "${payload.argument}" will be provided once the LLM is connected.`;
    case 'translate':
      return `[mock translate] Translation of "${payload.argument}" will be provided once the LLM is connected.`;
    case 'chat':
      return `[mock chat] You said: "${payload.argument}". The assistant is not yet connected.`;
    default:
      return '[mock] Unsupported command.';
  }
}

/**
 * Build a WeChat-compliant text reply XML envelope.
 * Note: ToUserName/FromUserName are swapped relative to the incoming message.
 */
function buildTextReply(toUser: string, fromUser: string, content: string): string {
  const createTime = Math.floor(Date.now() / 1000);
  return `<xml>
<ToUserName><![CDATA[${toUser}]]></ToUserName>
<FromUserName><![CDATA[${fromUser}]]></FromUserName>
<CreateTime>${createTime}</CreateTime>
<MsgType><![CDATA[text]]></MsgType>
<Content><![CDATA[${content}]]></Content>
</xml>`;
}

// ---------- Routes ----------

// GET /wechat - WeChat server URL verification handshake
app.get('/wechat', (req: Request, res: Response): void => {
  const { signature, timestamp, nonce, echostr } = req.query;

  if (
    typeof signature !== 'string' ||
    typeof timestamp !== 'string' ||
    typeof nonce !== 'string' ||
    typeof echostr !== 'string'
  ) {
    res.status(400).send('Bad Request');
    return;
  }

  if (verifyWeChatSignature(WECHAT_TOKEN, timestamp, nonce, signature)) {
    res.status(200).send(echostr);
    return;
  }

  res.status(401).send('Unauthorized');
});

// POST /wechat - Incoming user messages
app.post('/wechat', async (req: Request, res: Response): Promise<void> => {
  const xmlBody = req.body;

  if (typeof xmlBody !== 'string' || xmlBody.length === 0) {
    res.status(400).send('Bad Request');
    return;
  }

  try {
    const parsed = (await parseStringPromise(xmlBody, {
      explicitArray: false,
      trim: true,
    })) as ParsedWeChatEnvelope;

    const message = parsed.xml;

    // Only text messages are supported in this initial version
    if (message.MsgType !== 'text' || !message.Content) {
      res.status(200).send('success');
      return;
    }

    const { type, argument } = routeCommand(message.Content);

    const replyContent = await processCommand(type, {
      fromUser: message.FromUserName,
      toUser: message.ToUserName,
      argument,
    });

    const replyXml = buildTextReply(message.FromUserName, message.ToUserName, replyContent);

    res.set('Content-Type', 'application/xml');
    res.status(200).send(replyXml);
  } catch (err) {
    console.error('Failed to handle WeChat message:', err);
    // Respond with 'success' so the WeChat server does not retry indefinitely
    res.status(200).send('success');
  }
});

// ---------- Server ----------

app.listen(PORT, () => {
  console.log(`WeChat AI assistant listening on port ${PORT}`);
});
