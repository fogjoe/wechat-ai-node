import express, { Request, Response } from 'express';
import crypto from 'crypto';
import { parseStringPromise } from 'xml2js';
import dotenv from 'dotenv';
import OpenAI from 'openai';

dotenv.config();

const PORT: number = parseInt(process.env.PORT ?? '3006', 10);
const WECHAT_TOKEN: string = process.env.WECHAT_TOKEN ?? '';
const OPENROUTER_API_KEY: string = process.env.OPENROUTER_API_KEY ?? '';
const VOCAB_SYSTEM_PROMPT = `You are a trilingual vocabulary assistant.
The user will give you a word or phrase in any language.
Analyze the exact input term as a complete lexical item. Do not replace it with a synonym, root word, component character, or related adjective.
If the input is a compound word or phrase, define the whole compound or phrase.
For example, "美女" means "beautiful woman"; do not answer as "美丽" or "beautiful".
Reply strictly in this format:
ES: <Spanish equivalent> [IPA or pronunciation]
Def: <one concise Spanish definition>
Ej: <one short Spanish example sentence>

EN: <English equivalent> [IPA or pronunciation]
Def: <one concise English definition>
Ex: <one short English example sentence>

ZH: <Chinese equivalent> (<pinyin>)
释义: <one concise Chinese definition>
例句: <one short Chinese example sentence>

Keep the whole reply concise. Do not add markdown, numbering, explanations, or extra labels.`;

if (!WECHAT_TOKEN) {
  console.error('Missing WECHAT_TOKEN environment variable.');
  process.exit(1);
}

if (!OPENROUTER_API_KEY) {
  console.warn('OPENROUTER_API_KEY is not set; LLM features will be unavailable.');
}

const app = express();
const openai = new OpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: OPENROUTER_API_KEY || 'missing-openrouter-api-key',
  timeout: 4500,
  defaultHeaders: {
    'HTTP-Referer': 'http://localhost:3006',
    'X-Title': 'WeChat-Mac-Mini-Bot',
  },
});

app.use((req, _res, next) => {
  console.log(`Incoming request: ${req.method} ${req.originalUrl}`);
  next();
});

// Parse raw bodies defensively because WeChat gateways may vary content types.
app.use(express.text({ type: '*/*', limit: '1mb' }));

// ---------- Types ----------

interface WeChatTextMessage {
  ToUserName: string;
  FromUserName?: string;
  CreateTime: string;
  MsgType?: string;
  Content?: string;
  MsgId?: string;
  Encrypt?: string;
}

interface ParsedWeChatEnvelope {
  xml: WeChatTextMessage;
}

interface ReplyTarget {
  toUser: string;
  fromUser: string;
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
 * Build a WeChat-compliant text reply XML envelope.
 * Note: ToUserName/FromUserName are swapped relative to the incoming message.
 */
function buildTextReply(toUser: string, fromUser: string, content: string): string {
  const createTime = Math.floor(Date.now() / 1000);
  const cdataContent = content.replaceAll(']]>', ']]]]><![CDATA[>');

  return `<xml>
<ToUserName><![CDATA[${toUser}]]></ToUserName>
<FromUserName><![CDATA[${fromUser}]]></FromUserName>
<CreateTime>${createTime}</CreateTime>
<MsgType><![CDATA[text]]></MsgType>
<Content><![CDATA[${cdataContent}]]></Content>
</xml>`;
}

function extractReplyTarget(xmlBody: string): ReplyTarget | null {
  const fromUser = xmlBody.match(/<FromUserName><!\[CDATA\[(.*?)\]\]><\/FromUserName>/)?.[1];
  const toUser = xmlBody.match(/<ToUserName><!\[CDATA\[(.*?)\]\]><\/ToUserName>/)?.[1];

  if (!fromUser || !toUser) {
    return null;
  }

  return {
    toUser: fromUser,
    fromUser: toUser,
  };
}

function sanitizeVocabularyTerm(term: string): string {
  return term.trim().replace(/^[\s"'“”‘’.,!?;:，。！？；：、]+|[\s"'“”‘’.,!?;:，。！？；：、]+$/g, '');
}

async function lookupVocabulary(term: string): Promise<string> {
  if (!OPENROUTER_API_KEY) {
    return 'OpenRouter API key is not configured.';
  }

  const completion = await openai.chat.completions.create({
    model: 'openai/gpt-4o-mini',
    temperature: 0,
    max_tokens: 360,
    messages: [
      {
        role: 'system',
        content: VOCAB_SYSTEM_PROMPT,
      },
      {
        role: 'user',
        content: `Exact term: ${term}`,
      },
    ],
  });

  return completion.choices[0]?.message?.content?.trim() || 'No response was returned.';
}

async function createReplyContent(content: string): Promise<string> {
  const normalizedContent = content.trim();
  const vocabMatch = normalizedContent.match(/^#vocab\s+(.+)$/s);

  if (!vocabMatch) {
    return 'Commands available: #vocab [word]';
  }

  const term = sanitizeVocabularyTerm(vocabMatch[1] ?? '');
  if (!term) {
    return 'Commands available: #vocab [word]';
  }

  try {
    return await lookupVocabulary(term);
  } catch (err) {
    console.error('OpenRouter request failed:', err);
    return 'OpenRouter request failed. Please try again later.';
  }
}

// ---------- Routes ----------

app.get('/', (_req: Request, res: Response): void => {
  res.status(200).send('WeChat AI assistant is running.');
});

app.get('/health', (_req: Request, res: Response): void => {
  res.status(200).json({
    ok: true,
    port: PORT,
    openRouterConfigured: Boolean(OPENROUTER_API_KEY),
  });
});

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
    console.warn('Received WeChat POST with an empty or unparsed body.');
    res.status(400).send('Bad Request');
    return;
  }

  console.log(`Received WeChat POST body length: ${xmlBody.length}`);
  const replyTarget = extractReplyTarget(xmlBody);

  try {
    const parsed = (await parseStringPromise(xmlBody, {
      explicitArray: false,
      trim: true,
    })) as ParsedWeChatEnvelope;

    const message = parsed.xml;
    console.log(`Received WeChat message type: ${message.MsgType}`);

    if (message.Encrypt && !message.MsgType) {
      console.error(
        'Received an encrypted WeChat message. Configure WeChat message encryption as plaintext or add AES decryption support.',
      );
      res.status(200).send('success');
      return;
    }

    if (!message.FromUserName || !message.ToUserName) {
      console.error('WeChat message is missing FromUserName or ToUserName.');
      res.status(200).send('success');
      return;
    }

    // Reply with XML for unsupported message types so WeChat does not retry.
    if (message.MsgType !== 'text' || !message.Content) {
      const replyXml = buildTextReply(message.FromUserName, message.ToUserName, 'success');

      res.set('Content-Type', 'application/xml');
      res.status(200).send(replyXml);
      return;
    }

    const replyContent = await createReplyContent(message.Content);
    const replyXml = buildTextReply(message.FromUserName, message.ToUserName, replyContent);

    res.set('Content-Type', 'application/xml');
    res.status(200).send(replyXml);
  } catch (err) {
    console.error('Failed to handle WeChat message:', err);

    if (replyTarget) {
      const replyXml = buildTextReply(
        replyTarget.toUser,
        replyTarget.fromUser,
        'Failed to process the message. Please try again later.',
      );

      res.set('Content-Type', 'application/xml');
      res.status(200).send(replyXml);
      return;
    }

    // Respond with success only when a valid reply target cannot be recovered.
    res.status(200).send('success');
  }
});

app.use((req: Request, res: Response): void => {
  console.warn(`No route matched: ${req.method} ${req.originalUrl}`);
  res.status(404).send('Not Found');
});

// ---------- Server ----------

app.listen(PORT, () => {
  console.log(`WeChat AI assistant listening on port ${PORT}`);
});
