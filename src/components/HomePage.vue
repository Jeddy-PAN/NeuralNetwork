<template>
  <div class="converter">
    <!-- 连接切换器 -->
    <div class="connect-switcher">
      <label class="switch">
        <input 
          type="checkbox" 
          v-model="isConnected" 
          @change="handleConnectToggle"
        >
        <span class="slider"></span>
      </label>
      <span class="status-label">{{ isConnected ? 'Connected' : 'Disconnected' }}</span>
    </div>

    <!-- 连接后功能区 -->
    <div v-if="isConnected" class="connected-functions">
      <!-- ID 功能区 -->
      <div class="id-section">
        <button @click="handleGetId" class="btn btn-primary">GET ID</button>
        <div class="id-display">{{ client_id || 'NO ID' }}</div>
      </div>

      <!-- Ready 状态区 -->
      <div class="ready-section">
        <button 
          @click="toggleReady"
          class="btn btn-ready"
          :class="{ disabled: isReady }"
          :disabled="isReady"
        >
          {{ isReady ? 'IS READY' : 'GET READY' }}
        </button>
        <div class="status-indicator">
          <div class="dot" :class="{ ready: isReady }"></div>
          <span>{{ isReady ? 'IS READY' : 'NOT READY' }}</span>
        </div>
      </div>

      <!-- Start 控制区 -->
      <div class="start-control">
        <div class="options">
          <select v-model="startOption1">
            <option value="option1">OPTION1</option>
            <option value="option2">OPTION2</option>
          </select>
          <select v-model="startOption2">
            <option value="modeA">MODEA</option>
            <option value="modeB">MODEB</option>
          </select>
        </div>
        <button @click="handleStart" class="btn btn-start">START</button>
      </div>

      <!-- Stop 按钮 -->
      <button @click="handleStop" class="btn btn-stop">STOP</button>

      <!-- Reset/Clear 按钮 -->
      <div class="reset-section">
        <button @click="handleReset" class="btn btn-reset">RESET</button>
        <button @click="handleClear" class="btn btn-clear">CLEAR</button>
      </div>
    </div>
  </div>
  <ClassifyPlot :reset-flag="resetPlotFlag" @classifyResetComplete="resetPlotFlag = false" />
  <LossPlot :reset-flag="resetPlotFlag" @resetComplete="resetPlotFlag = false" />
</template>

<script setup>
import { ref } from 'vue';
import { wsManager } from '../utils/backend/CPU/tools/websocketManager';
import { 
    getClientId,
    setReadyForTrain, 
    resetServer
} from '../utils/backend/CPU/tools/client';
import { setFlagTrain, setFlagStop } from '../utils/backend/GPU/initModel/GPUTraining';
import ClassifyPlot from './ClassifyPlot.vue';
import LossPlot from './LossPlot.vue';
import { startTrain } from '../utils/backend/CPU/ModelSetup/setUpData';


// 状态管理
const isConnected = ref(false);
const isReady = ref(false);
const client_id = ref('');
const ws_id = ref('');
const startOption1 = ref('option1');
const startOption2 = ref('modeA');
const resetPlotFlag = ref(false);

// 连接/断开切换
const handleConnectToggle = async() => {
  if (isConnected.value) {
    ws_id.value = await getClientId();
    try {
			await wsManager.connect(ws_id.value);
			console.log('WebSocket connection established');
		} catch (error) {
			console.warn('WebSocket connection failed, will use polling fallback:', error);
	}
  } else {
    if(wsManager.isConnected) {
        wsManager.disconnect();
        isReady.value = false;
        ws_id.value = '';
        client_id.value = '';
    }   
  }
};

// 其他功能
const toggleReady = async() => {
    await setReadyForTrain(client_id.value);
    isReady.value = true;
}

const handleGetId = () => {
    client_id.value = ws_id.value;
}

const handleStart = async() => {
    setFlagTrain();
    startTrain();
}

const handleStop = () => {
    setFlagStop();
}

const handleReset = async() => {
    await resetServer();
    isReady.value = false;
    resetPlotFlag.value = true;
}

const handleClear = () => { 
    localStorage.clear();
    client_id.value = '';
};
</script>

<style scoped>
/* 基础容器 */
.converter {
  max-width: 420px;
  margin: 0 auto;
  padding: 16px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
}

/* 连接切换器 */
.connect-switcher {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
}

.switch {
  position: relative;
  display: inline-block;
  width: 50px;
  height: 24px;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #ccc;
  transition: .4s;
  border-radius: 24px;
}

.slider:before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: .4s;
  border-radius: 50%;
}

input:checked + .slider {
  background-color: #42b983;
}

input:checked + .slider:before {
  transform: translateX(26px);
}

.status-label {
  margin-left: 12px;
  font-weight: 500;
}

/* 连接后功能区 */
.connected-functions {
  display: grid;
  gap: 12px;
}

/* ID 功能区 */
.id-section {
  display: flex;
  gap: 8px;
  align-items: center;
}

.id-display {
  padding: 8px 12px;
  background-color: #f5f5f5;
  border-radius: 4px;
  min-width: 120px;
}

/* Ready 状态区 */
.ready-section {
  display: flex;
  gap: 8px;
  align-items: center;
}

/* 按钮基础样式 */
.btn {
  padding: 8px 16px;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.btn-primary {
  background-color: #646cff;
}

.btn-ready {
  background-color: #f0ad4e;
}

.btn-ready.disabled {
  background-color: #aaa;
  cursor: default;
}

.btn-start {
  background-color: #28a745;
  width: 100%;
}

.btn-stop {
  background-color: #dc3545;
}

.btn-reset {
  background-color: #6c757d;
}

.btn-clear {
  background-color: #343a40;
}

/* 状态指示器 */
.status-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
}

.dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: #999;
}

.dot.ready {
  background-color: #42b983;
}

/* Start 控制区 */
.start-control {
  padding: 12px;
  background-color: white;
  border: 1px solid #ddd;
  border-radius: 6px;
}

.options {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

select {
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  flex: 1;
  min-width: 120px;
}

/* Reset/Clear 按钮 */
.reset-section {
  display: flex;
  gap: 8px;
}

.reset-section .btn {
  flex: 1;
}
</style>
