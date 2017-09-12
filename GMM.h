#pragma once
#include <Eigen/Dense>
#define THRESHOLD 1e-15//两次迭代的阈值;
#define PI 3.1415926535898
#define TRAINRATE 3//总样本数是训练样本个数;

class GMM
{
public:
	GMM(int n,int dim);
	~GMM();
	void practiceModel();
	void initParams();
	void classify();
	void setData(Eigen::MatrixXd* data)
	{
		m_data = data;
	}
	void output();
	//计算data矩阵中每个像素点属于该类的概率;
	//data为nData*dim的矩阵;
	//probability为nData*nClassify的矩阵，此函数修改probability矩阵的第n列;
	//mode为计算模式，1为整体进行，2为分部进行;
	void calculateProbility(Eigen::MatrixXd* data, Eigen::MatrixXd* probability,int n);
	//每次迭代计算的迭代值;
	Eigen::MatrixXd* GMM::calcProb();
	//对in矩阵求协方差，其中每一项为列向量;
	//n为正在对第几个高斯模型求解协方差矩阵;
	void cov(Eigen::MatrixXd* in,Eigen::MatrixXd* out);
private:
	int m_n;//含有的高斯模型个数;
	Eigen::MatrixXd *m_miu;//每个高斯模型的期望值;
	int m_dim;//数据的维度;
	Eigen::MatrixXd **m_omega;//每个高斯模型的协方差矩阵;
	Eigen::MatrixXd *m_pai;//每个高斯模型的权值;
	Eigen::MatrixXd* m_data;//用于模型训练的矩阵;
	int n_data;
};

