#pragma once
#include <Eigen/Dense>
#define THRESHOLD 1e-15//���ε�������ֵ;
#define PI 3.1415926535898
#define TRAINRATE 3//����������ѵ����������;

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
	//����data������ÿ�����ص����ڸ���ĸ���;
	//dataΪnData*dim�ľ���;
	//probabilityΪnData*nClassify�ľ��󣬴˺����޸�probability����ĵ�n��;
	//modeΪ����ģʽ��1Ϊ������У�2Ϊ�ֲ�����;
	void calculateProbility(Eigen::MatrixXd* data, Eigen::MatrixXd* probability,int n);
	//ÿ�ε�������ĵ���ֵ;
	Eigen::MatrixXd* GMM::calcProb();
	//��in������Э�������ÿһ��Ϊ������;
	//nΪ���ڶԵڼ�����˹ģ�����Э�������;
	void cov(Eigen::MatrixXd* in,Eigen::MatrixXd* out);
private:
	int m_n;//���еĸ�˹ģ�͸���;
	Eigen::MatrixXd *m_miu;//ÿ����˹ģ�͵�����ֵ;
	int m_dim;//���ݵ�ά��;
	Eigen::MatrixXd **m_omega;//ÿ����˹ģ�͵�Э�������;
	Eigen::MatrixXd *m_pai;//ÿ����˹ģ�͵�Ȩֵ;
	Eigen::MatrixXd* m_data;//����ģ��ѵ���ľ���;
	int n_data;
};

