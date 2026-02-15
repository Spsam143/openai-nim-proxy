const express = require('express');
const axios = require('axios');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '50mb' }));

const NIM_API_KEY = process.env.NIM_API_KEY;
const NIM_BASE_URL = 'https://integrate.api.nvidia.com/v1/chat/completions';
const SHOW_REASONING = false;

const MODEL_MAPPING = {
  'gpt-4o': 'z-ai/glm5',
  'gpt-4': 'deepseek-ai/deepseek-v3.1',
  'gpt-4-turbo': 'moonshotai/kimi-k2.5-v1',
  'gpt-4-32k': 'qwen/qwen3-235b-a22b',
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'claude-3-opus': 'qwen/qwen3-coder-480b-a35b-instruct',
  'claude-3.5-sonnet': 'qwen/qwen3-235b-a22b-instruct',
  'claude-3-haiku': 'stepfun/stepfun-3.5-flash',
  'gemini-exp-1206': 'meta/llama-3.1-405b-instruct',
  'gemini-2.0-flash': 'minimax/minimax-m2.1',
  'o1-preview': 'deepseek-ai/deepseek-r1',
  'o1': 'deepseek-ai/deepseek-r1-distill-llama-70b',
  'o3-mini': 'qwen/qwq-32b-preview'
};

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING ? 'enabled' : 'disabled',
    thinking_mode: false
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature = 0.7, max_tokens, stream = false } = req.body;
    const nimModel = MODEL_MAPPING[model] || model;

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature,
      max_tokens: max_tokens || 4096,
      stream: false
    };

    const response = await axios.post(NIM_BASE_URL, nimRequest, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${NIM_API_KEY}`
      }
    });

    const nimResponse = response.data;
    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: [{
        index: 0,
        message: nimResponse.choices[0].message,
        finish_reason: nimResponse.choices[0].finish_reason
      }],
      usage: nimResponse.usage
    };

    res.json(openaiResponse);
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.response?.data?.detail || error.message,
        type: 'proxy_error'
      }
    });
  }
});

app.listen(PORT, () => {
  console.log(`Proxy running on port ${PORT}`);
});
