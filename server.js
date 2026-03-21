const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors({ origin: '*', credentials: false, methods: ['GET', 'POST', 'OPTIONS'], allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'] }));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = true;

const MODEL_MAPPING = {
  'gpt-4o':           'deepseek-ai/deepseek-v3-1',
  'gpt-4':            'deepseek-ai/deepseek-r1',
  'gpt-4-turbo':      'deepseek-ai/deepseek-v3-1-instruct',
  'gpt-4-32k':        'deepseek-ai/deepseek-v3.2',
  'gpt-3.5-turbo-16k':'deepseek-ai/deepseek-r1-distill-qwen-32b',
  'gpt-3.5-turbo':    'moonshotai/kimi-k2.5',
  'claude-3-haiku':   'moonshotai/kimi-k2-thinking',
  'claude-3-opus':    'mistral-ai/mistral-large-3-675b-instruct-2512',
  'claude-3-sonnet':  'minimaxai/minimax-m2.5',
  'gemini-pro':       'z-ai/glm-5-thinking',
  'gemini-pro-vision':'z-ai/glm4.7'
};

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to NVIDIA NIM Proxy', reasoning_display: SHOW_REASONING, thinking_mode: ENABLE_THINKING_MODE });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({ id: model, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy' }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    let nimModel = MODEL_MAPPING[model] || model;

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };

    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningBuffer = '';
      let reasoningSent = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          if (line.includes('[DONE]')) { res.write(line + '\n'); return; }

          try {
            const data = JSON.parse(line.slice(6));
            if (data.choices?.[0]?.delta) {
              const reasoning = data.choices[0].delta.reasoning_content || '';
              const content = data.choices[0].delta.content || '';

              delete data.choices[0].delta.reasoning_content;

              if (SHOW_REASONING && reasoning) {
                reasoningBuffer += reasoning;
                data.choices[0].delta.content = '';
                res.write(`data: ${JSON.stringify(data)}\n\n`);
                return;
              }

              if (SHOW_REASONING && content && !reasoningSent && reasoningBuffer) {
                const thinkBlock = `<think>${reasoningBuffer}</think>\n\n${content}`;
                data.choices[0].delta.content = thinkBlock;
                reasoningSent = true;
                reasoningBuffer = '';
                res.write(`data: ${JSON.stringify(data)}\n\n`);
                return;
              }

              data.choices[0].delta.content = content;
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            }
          } catch (e) { res.write(line + '\n'); }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', () => res.end());

    } else {
      let fullContent = response.data.choices[0]?.message?.content || '';
      const reasoning = response.data.choices[0]?.message?.reasoning_content || '';

      if (SHOW_REASONING && reasoning) {
        fullContent = `<think>${reasoning}</think>\n\n${fullContent}`;
      }

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          message: { role: 'assistant', content: fullContent },
          finish_reason: response.data.choices[0]?.finish_reason
        }],
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }

  } catch (error) {
    res.status(error.response?.status || 500).json({ error: { message: error.message || 'Internal server error', type: 'invalid_request_error', code: error.response?.status || 500 } });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
