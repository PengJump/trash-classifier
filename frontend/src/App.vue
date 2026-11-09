<script setup>
import { ref } from 'vue'
import axios from 'axios'

const file = ref(null)
const preview = ref('')
const result = ref(null)
const loading = ref(false)
const error = ref('')
const CLASS_KEYS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
const CLASS_CN = ['纸箱', '玻璃', '金属', '纸类', '塑料', '其他垃圾']

function onFile(e) {
  const f = e.target.files[0]
  if (!f) return
  file.value = f
  preview.value = URL.createObjectURL(f)
  result.value = null
  error.value = ''
}

async function upload() {
  if (!file.value) {
    error.value = '先选一张图'
    return
  }
  loading.value = true
  error.value = ''
  try {
    const fd = new FormData()
    fd.append('file', file.value)
    const { data } = await axios.post('/api/classify', fd)
    result.value = data
  } catch (e) {
    error.value = '识别失败：' + (e.response?.data?.message || e.message)
  } finally {
    loading.value = false
  }
}

function percent(x) {
  return Math.round(x * 100) + '%'
}

function pick(i) {
  result.value = {
    ...result.value,
    label: CLASS_KEYS[i],
    labelCn: CLASS_CN[i],
    category: '人工修正',
    tip: '已按你的选择记录，作为纠错样本'
  }
}
</script>

<template>
  <div class="wrap">
    <h1>这袋垃圾，往哪扔？</h1>
    <p class="sub">拍照识别可回收物，给你投放提示</p>

    <div class="upload">
      <input type="file" accept="image/*" @change="onFile" />
      <img v-if="preview" :src="preview" class="preview" alt="预览" />
      <button :disabled="!file || loading" @click="upload">
        {{ loading ? '识别中…' : '识别' }}
      </button>
    </div>

    <p v-if="error" class="err">{{ error }}</p>

    <div v-if="result" class="card">
      <div class="row">
        <span class="label">{{ result.labelCn }}</span>
        <span class="conf">{{ percent(result.confidence) }}</span>
      </div>
      <div class="bar"><div class="fill" :style="{ width: percent(result.confidence) }"></div></div>
      <p class="cat">{{ result.category }}</p>
      <p class="tip">{{ result.tip }}</p>

      <div v-if="result.label === 'uncertain'" class="correct">
        <p>看着像哪一类？手动选：</p>
        <div class="btns">
          <button v-for="(k, i) in CLASS_KEYS" :key="k" @click="pick(i)">
            {{ CLASS_CN[i] }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style>
body {
  margin: 0;
  background: #f6f7f9;
  font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  color: #1c2733;
}
.wrap { max-width: 560px; margin: 0 auto; padding: 32px 20px; }
h1 { font-size: 26px; margin: 0 0 4px; }
.sub { color: #667; margin: 0 0 20px; }
.upload { display: flex; flex-direction: column; gap: 12px; }
.preview { max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }
button { padding: 10px 18px; border: none; border-radius: 6px; background: #1f5f73; color: #fff; font-size: 15px; cursor: pointer; }
button:disabled { background: #9fb3bb; cursor: not-allowed; }
.err { color: #c0392b; }
.card { margin-top: 20px; background: #fff; border: 1px solid #e2e6ea; border-radius: 10px; padding: 20px; }
.row { display: flex; justify-content: space-between; align-items: baseline; }
.label { font-size: 24px; font-weight: 700; }
.conf { font-size: 18px; color: #1f5f73; font-weight: 700; }
.bar { height: 8px; background: #eef1f4; border-radius: 4px; margin: 10px 0; overflow: hidden; }
.fill { height: 100%; background: #1f5f73; }
.cat { font-size: 15px; color: #d9683c; font-weight: 600; margin: 8px 0 4px; }
.tip { font-size: 15px; margin: 0; }
.correct { margin-top: 14px; border-top: 1px dashed #ddd; padding-top: 12px; }
.correct p { margin: 0 0 8px; color: #667; }
.btns { display: flex; flex-wrap: wrap; gap: 8px; }
.btns button { background: #fff; color: #1f5f73; border: 1px solid #1f5f73; }
</style>
