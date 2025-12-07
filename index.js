const app = getApp();
const { getConstitutionList } = require('../../utils/data.js');

Page({
  data: {
    constitutionList: [],
    selectedId: null
  },

  onLoad() {
    // 获取体质列表数据
    const list = getConstitutionList();
    this.setData({
      constitutionList: list
    });
  },

  // 选择体质
  selectConstitution(e) {
    const id = e.currentTarget.dataset.id;
    this.setData({
      selectedId: id
    });
  },

  // 提交选择
  submitSelection() {
    if (!this.data.selectedId) return;
    
    wx.navigateTo({
      url: `/pages/result/result?id=${this.data.selectedId}`
    });
  }
});