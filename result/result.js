// 修改后的 result.js
const app = getApp();
const { getConstitutionById, getFoodsByConstitutionId, getNutritionByFoodId } = require('../../utils/data.js');

Page({
  onNavigateBack() {
    wx.reLaunch({ url: '/pages/index/index' });
  },

  data: {
    constitutionId: null,
    constitutionName: '',
    foodList: []
  },

  onLoad(options) {
    const id = parseInt(options.id);
    this.setData({ constitutionId: id });

    const constitution = getConstitutionById(id);
    if (constitution) {
      this.setData({ constitutionName: constitution.name });
    }

    this.getRecommendedFoods(id);
  },

  getRecommendedFoods(constitutionId) {
    const foods = getFoodsByConstitutionId(constitutionId);
    console.log('【1】推荐食物', foods); // ★ 看有没有食物
  
    foods.forEach(f => {
      console.log('【2】食物 ID', f.identity, f.name);
      const nutrients = getNutritionByFoodId(f.identity);
      console.log('【3】营养素数组', nutrients); // ★ 看返回什么
    });
    
    if (!foods.length) return;
  
    const containerW = 750;
    const containerH = 500;
    const cx = containerW / 2;
    const cy = containerH / 2;
    const centerCircleRadius = 100;
    const foodNodeRadius = 60;
    const minOrbitRadius = 200;
    
    const orbitRadius = Math.max(minOrbitRadius, 150 + foods.length * 30);
    
    const positionedFoods = foods.map((food, idx) => {
      const angle = (idx / foods.length) * 2 * Math.PI;
      const nodeX = cx + orbitRadius * Math.cos(angle);
      const nodeY = cy + orbitRadius * Math.sin(angle);
      
      const dx = nodeX - cx;
      const dy = nodeY - cy;
      const lineAngle = Math.atan2(dy, dx) * 180 / Math.PI;
      
      const distance = Math.sqrt(dx * dx + dy * dy);
      const lineLength = distance - centerCircleRadius - foodNodeRadius;
      
      const nutrients = getNutritionByFoodId(food.identity);
      
      // 添加调试信息
      console.log('【4】最终食物对象', {
        ...food,
        nutrients: nutrients,
        left: (nodeX - foodNodeRadius).toFixed(0),
        top: (nodeY - foodNodeRadius).toFixed(0),
        lineLength: Math.max(40, lineLength).toFixed(0),
        lineAngle: lineAngle.toFixed(2),
        lineStartX: (cx + (centerCircleRadius * Math.cos(angle))).toFixed(0),
        lineStartY: (cy + (centerCircleRadius * Math.sin(angle))).toFixed(0)
      });
      
      return {
        ...food,
        nutrients: nutrients,
        left: (nodeX - foodNodeRadius).toFixed(0),
        top: (nodeY - foodNodeRadius).toFixed(0),
        lineLength: Math.max(40, lineLength).toFixed(0),
        lineAngle: lineAngle.toFixed(2),
        lineStartX: (cx + (centerCircleRadius * Math.cos(angle))).toFixed(0),
        lineStartY: (cy + (centerCircleRadius * Math.sin(angle))).toFixed(0)
      };
    });
    
    console.log('【5】最终食物列表', positionedFoods);
    this.setData({ foodList: positionedFoods });
  }
});