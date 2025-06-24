<template>
  <div class="converter">
    <div class="connect-switcher">
      <label class="switch">
        <input 
          type="checkbox" 
          v-model="isConnected" 
          @change="handleConnectToggle"
        >
        <span class="slider"></span>
      </label>
      <span class="status-text" style="margin-left: 10px;">Connect Server</span>
    </div>
    <!-- 连接状态显示 -->
    <div class="connection-status">
      <div class="status-indicator">
        <span class="status-dot" :class="{ 'connected': isConnected, 'disconnected': !isConnected }"></span>
        <span class="status-text">{{ isConnected ? 'WebSocket Connected' : 'WebSocket Disconnected' }}</span>
      </div>
      <div v-if="ws_id" class="id-display">ID: {{ ws_id }}</div>
    </div>

    <!-- 训练控制区 -->
    <div class="training-controls">
      <!-- 数据集选择 -->
      <div class="dataset-selection">
        <label>Dataset:</label>
        <select v-model="dataOption">
          <option value="dataClass1.csv">dataClass1.csv</option>
          <option value="dataClass2.csv">dataClass2.csv</option>
          <option value="dataClass3.csv">dataClass3.csv</option>
        </select>
      </div>

      <!-- 训练按钮 -->
      <button 
        @click="handleStart" 
        :disabled="isTraining"
        class="btn btn-start"
      >
        {{ isTraining ? 'Training...' : 'START TRAINING' }}
      </button>

      <!-- 停止按钮 -->
      <button 
        @click="handleStop" 
        :disabled="!isTraining"
        class="btn btn-stop"
      >
        STOP TRAINING
      </button>

      <!-- 重置按钮 -->
      <div class="reset-section">
        <button @click="handleReset" class="btn btn-reset">RESET</button>
        <button @click="handleClear" class="btn btn-clear">CLEAR ALL</button>
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
    resetServer,
} from '../utils/backend/CPU/tools/client';
import { setFlagTrain, setFlagStop } from '../utils/backend/GPU/initModel/GPUTraining';
import ClassifyPlot from './ClassifyPlot.vue';
import LossPlot from './LossPlot.vue';
import { startTrain } from '../utils/backend/CPU/tools/client';

// 状态管理
const isConnected = ref(false);
const isTraining = ref(false);
const ws_id = ref('');
const dataOption = ref('dataClass1.csv');
const resetPlotFlag = ref(false);

const handleConnectToggle = async() => {
  if(isConnected.value) {
    // 获取客户端ID
    ws_id.value = await getClientId();
    console.log('Client ID obtained:', ws_id.value);
        
    // 连接WebSocket
    await wsManager.connect(ws_id.value);
    isConnected.value = true;
    console.log('WebSocket connection established');
  } else {
      if (wsManager.isConnected()) {
        await handleStop();
        wsManager.disconnect();
        ws_id.value = '';
        console.log('WebSocket disconnected');
      }
      isConnected.value = false;
  }
}
// 开始训练
const handleStart = async() => {
    if (!isConnected.value) {
      console.error('Not connected to the Server');
      return;
    }
    if (isTraining.value) {
        console.warn('Training is already in progress');
        return;
    }
    
    try {
        isTraining.value = true;
        
        // 先重置图
        resetPlotFlag.value = true;
        
        // 开始训练
        setFlagTrain();
        console.log('Starting training with dataset:', dataOption.value);
        await startTrain(dataOption.value);
        
    } catch (error) {
        console.error('Failed to start training:', error);
        isTraining.value = false;
        alert('Failed to start training: ' + error.message);
    }
};

// 停止训练
const handleStop = async() => {
    if (!isConnected.value) {
      console.error('Not connected to the Server');
      return;
    }
    if (!isTraining.value) {
        console.warn('No training in progress');
        return;
    }
    
    try {
        // 停止训练标志
        setFlagStop();
        console.log('Training stopped');
        
        // // 断开WebSocket连接
        // if (wsManager.isConnected()) {
        //     wsManager.disconnect();
        //     console.log('WebSocket disconnected');
        // }
        isTraining.value = false;
        // isConnected.value = false;
        
    } catch (error) {
        console.error('Error stopping training:', error);
    }
};

// 重置服务器
const handleReset = async() => {
    if (!isConnected.value) {
      console.error('Not connected to the Server');
      return;
    }
    try {
        await resetServer();
        resetPlotFlag.value = true;
        console.log('Server reset completed');
    } catch (error) {
        console.error('Failed to reset server:', error);
    }
};

// 清除所有数据
const handleClear = async() => {
    if (!isConnected.value) {
      console.error('Not connected to the Server');
      return;
    }
    try {
        // 先停止训练
        if (isTraining.value) {
            await handleStop();
        }
        
        // 重置服务器
        await handleReset();
        
        // 清除本地存储
        localStorage.clear();
        ws_id.value = '';
        // isConnected.value = false;
        isTraining.value = false;
        
        console.log('All data cleared');
    } catch (error) {
        console.error('Failed to clear data:', error);
    }
};
</script>

<style scoped>
/* 基础容器 */
.converter {
  max-width: 450px;
  margin: 0 auto;
  padding: 20px;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  background-color: #fafafa;
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

/* 连接状态显示 */
.connection-status {
  margin-bottom: 20px;
  padding: 12px;
  background-color: white;
  border-radius: 8px;
  border: 1px solid #ddd;
}

.status-indicator {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 8px;
  transition: background-color 0.3s;
}

.status-dot.connected {
  background-color: #28a745;
}

.status-dot.disconnected {
  background-color: #dc3545;
}

.status-text {
  font-weight: 500;
}

.id-display {
  font-size: 12px;
  color: #666;
  font-family: monospace;
}

/* 训练控制区 */
.training-controls {
  display: grid;
  gap: 16px;
}

/* 数据集选择 */
.dataset-selection {
  display: flex;
  align-items: center;
  gap: 8px;
}

.dataset-selection label {
  font-weight: 500;
  min-width: 70px;
}

.dataset-selection select {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background-color: white;
}

/* 按钮基础样式 */
.btn {
  padding: 12px 20px;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
  position: relative;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-start {
  background-color: #28a745;
  width: 100%;
}

.btn-start:hover:not(:disabled) {
  background-color: #218838;
}

.btn-stop {
  background-color: #dc3545;
  width: 100%;
}

.btn-stop:hover:not(:disabled) {
  background-color: #c82333;
}

.btn-reset {
  background-color: #6c757d;
}

.btn-reset:hover {
  background-color: #5a6268;
}

.btn-clear {
  background-color: #343a40;
}

.btn-clear:hover {
  background-color: #23272b;
}

/* Reset/Clear 按钮 */
.reset-section {
  display: flex;
  gap: 12px;
}

.reset-section .btn {
  flex: 1;
}
</style>
