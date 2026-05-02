const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors({
  origin: '*',
  credentials: false,
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.error('❌ Missing NIM_API_KEY environment variable');
  process.exit(1);
}

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;
const MAX_MESSAGES = 20;
const MAX_TOKENS = 4096;

const MODEL_MAPPING = {
  'gpt-4o': 'deepseek-ai/deepseek-v3-0324',
  'gpt-4': 'deepseek-ai/deepseek-r1',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3-0324',
  'gpt-4-32k': 'deepseek-ai/deepseek-v3-0324',
  'gpt-3.5-turbo-16k': 'google/gemma-4-27b-it',
  'gpt-3.5-turbo': 'google/gemma-4-27b-it',
  'claude-3-haiku': 'google/gemma-4-27b-it',
  'claude-3-opus': 'meta/llama-4-maverick-17b-128e-instruct',
  'claude-3-sonnet': 'minimaxai/minimax-m2.5',
  'gemini-pro': 'z-ai/glm-5.1',
  'gemini-pro-vision': 'z-ai/glm4.7'
};

function truncateMessages(messages = []) {
  if (!Array.isArray(messages)) return [];

  if (messages.length <= MAX_MESSAGES) return messages;

  const systemMsg = messages[0]?.role === 'system' ? [messages[0]] : [];
  const rest = messages.slice(-(MAX_MESSAGES - systemMsg.length));

  return [...systemMsg, ...rest];
}

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    max_messages: MAX_MESSAGES,
    max_tokens: MAX_TOKENS
  });
});

// Models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({ object: 'list', data: models });
});

// Chat completions
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const {
      model,
      messages = [],
      temperature = 0.7,
      max_tokens = MAX_TOKENS,
      stream = false
    } = req.body;

    if (!model) {
      return res.status(400).json({
        error: { message: 'Model is required', type: 'invalid_request_error' }
      });
    }

    const nimModel = MODEL_MAPPING[model] || model;
    const truncatedMessages = truncateMessages(messages);

    const nimRequest = {
      model: nimModel,
      messages: truncatedMessages,
      temperature,
      max_tokens,
      stream
    };

    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = {
        chat_template_kwargs: { thinking: true }
      };
    }

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 60000
      }
    );

    // STREAM MODE
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();

        let lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            return;
          }

          try {
            const parsed = JSON.parse(line.slice(6));

            if (parsed?.choices?.[0]?.delta) {
              const delta = parsed.choices[0].delta;

              // Clean unsupported field
              delete delta.reasoning_content;

              res.write(`data: ${JSON.stringify(parsed)}\n\n`);
            }
          } catch {
            // fallback raw line
            res.write(line + '\n\n');
          }
        }
      });

      response.data.on('end', () => res.end());
      response.data.on('error', () => res.end());

    } else {
      // NON-STREAM MODE
      const choice = response.data?.choices?.[0] || {};
      let content = choice?.message?.content || '';

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content
          },
          finish_reason: choice?.finish_reason || 'stop'
        }],
        usage: response.data?.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('❌ Error:', error?.response?.data || error.message);

    res.status(error.response?.status || 500).json({
      error: {
        message: error?.response?.data?.error?.message || error.message,
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// 404 fallback
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 NIM Proxy running on port ${PORT}`);
});