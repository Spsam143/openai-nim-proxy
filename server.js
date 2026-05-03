// server.js - OpenAI to NVIDIA NIM API Proxy (FULLY FIXED)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

const MODEL_MAPPING = {
  'gpt-4o'         : 'deepseek-ai/deepseek-v4-pro',
  'gpt-4-turbo'    : 'deepseek-ai/deepseek-v4-flash',
  'gpt-4'          : 'moonshotai/kimi-k2.6',
  'gpt-3.5-turbo'  : 'nvidia/nemotron-3-super-120b-a12b',
  'claude-3-opus'  : 'minimaxai/minimax-m2.7',
  'claude-3-sonnet': 'z-ai/glm-5.1',
  'gemini-pro'     : 'z-ai/glm-4.7',
  'claude-2'       : 'mistralai/mistral-medium-3.5-128b',
  'gpt-4-vision'   : 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  'gemini-ultra'   : 'mistralai/mistral-small-4-119b-2603'
};

// ✅ FIX 1: Retry helper with 3 attempts + 3s delay
async function fetchWithRetry(url, payload, headers, isStream, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      return await axios.post(url, payload, {
        headers,
        timeout: 300000, // ✅ FIX 2: 5 minute timeout (was missing)
        responseType: isStream ? 'stream' : 'json'
      });
    } catch (err) {
      if (i === retries - 1) throw err;
      console.log(`Retry ${i + 1}/${retries} after error: ${err.message}`);
      await new Promise(r => setTimeout(r, 3000));
    }
  }
}

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Main proxy endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      const modelLower = (model || '').toLowerCase();
      if (modelLower.includes('gpt-4') || modelLower.includes('405b')) {
        nimModel = 'meta/llama-3.1-405b-instruct';
      } else if (modelLower.includes('claude') || modelLower.includes('70b')) {
        nimModel = 'meta/llama-3.1-70b-instruct';
      } else {
        nimModel = 'meta/llama-3.1-8b-instruct';
      }
    }

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false,
      ...(ENABLE_THINKING_MODE && { extra_body: { chat_template_kwargs: { thinking: true } } })
    };

    // ✅ FIX 3: Use fetchWithRetry instead of raw axios.post
    const response = await fetchWithRetry(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      stream || false
    );

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        // ✅ FIX 4: Correct single \n (was broken as \\\\n)
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          if (line.includes('[DONE]')) {
            res.write(line + '\n\n');
            return;
          }
          try {
            const data = JSON.parse(line.slice(6));
            if (data.choices?.[0]?.delta) {
              const reasoning = data.choices[0].delta.reasoning_content;
              const content = data.choices[0].delta.content;

              if (SHOW_REASONING) {
                let combinedContent = '';
                if (reasoning && !reasoningStarted) {
                  combinedContent = '\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent = reasoning;
                }
                if (content && reasoningStarted) {
                  combinedContent += '\n\n' + content;
                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }
                if (combinedContent) {
                  data.choices[0].delta.content = combinedContent;
                  delete data.choices[0].delta.reasoning_content;
                }
              } else {
                data.choices[0].delta.content = content || '';
                delete data.choices[0].delta.reasoning_content;
              }
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });

    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '\n' + choice.message.reasoning_content + '\n\n\n' + fullContent;
          }
          return {
            index: choice.index,
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
