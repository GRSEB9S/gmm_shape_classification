#include "GMM.h"
#include <iostream>
#include<limits>
//#include <Eigen/src/Core/DiagonalMatrix.h>

GMM::GMM(int n, int dim) :m_n(n), m_dim(dim)
{
	m_miu = new Eigen::MatrixXd(m_n,m_dim);
	m_omega = new Eigen::MatrixXd*[n];
	//m_pai = new int[n];
	m_pai = new Eigen::MatrixXd(1,m_n);
	for (int i = 0; i < n;i++)
	{
		Eigen::MatrixXd* tmp = new Eigen::MatrixXd(m_dim, m_dim);
		*(m_omega + i) = tmp;
	}
}


GMM::~GMM()
{
	for (int i = 0; i < m_n;i++)
	{
		delete *(m_omega + i);
		//delete *(m_miu + i);
	}
	delete[]m_omega;
	delete m_miu;
	delete m_pai;
}

void GMM::practiceModel()
{
	n_data = m_data->rows();
	for (int i = 0; i < m_n;i++)
	{
		int randomN = rand() % (n_data-1);
		//std::cout << randomN << std::endl;
		m_miu->row(i) = m_data->row(randomN);
	}
	initParams();
	//double Lprev=std::numeric_limits<double>::min();
	double Lprev = -999999999;
	while (true)
	{
		//此处Px为n_data*m_n的矩阵;
		Eigen::MatrixXd* Px= calcProb();
		//Eigen::MatrixXd pGamma(n_data, m_n);
		Eigen::MatrixXd pGamma = Px->cwiseProduct(m_pai->replicate(n_data, 1));
		Eigen::MatrixXd pGammaSum= pGamma.rowwise().sum().replicate(1, m_n);
		pGamma = pGamma.eval().cwiseQuotient(pGammaSum);
		Eigen::VectorXd Nk = pGamma.colwise().sum();
		/**m_pai = Nk / n_data;
		Eigen::MatrixXd tmp= (pGamma.transpose()*(m_data->transpose())).transpose();
		for (int i = 0; i < m_n;i++)
		{
			m_miu->col(i) = (tmp.col(i) / Nk(i));
		}*/
		Eigen::MatrixXd dg = (1 / Nk.array()).matrix().asDiagonal();
		*m_miu = dg*pGamma.transpose()*(*m_data);
		*m_pai = Nk / n_data;
		m_pai->transposeInPlace();
		for (int i = 0; i < m_n;i++)
		{
			Eigen::MatrixXd XShift = (*m_data) - m_miu->row(i).replicate(n_data, 1);
			(*(m_omega[i])) = (XShift.transpose()*pGamma.col(i).asDiagonal()*XShift)
				/ Nk(i);
		}
		//std::cout << (*Px);// *(m_pai->transpose())).array().log();//.sum();
		//std::cout << (*Px) << std::endl;
		//std::cout << (*m_miu) << std::endl;
		double L = ((*Px)*(m_pai->transpose())).array().log().sum();
		delete Px;
		if (L-Lprev<THRESHOLD)
		{
			break;
		}
		Lprev = L;
	}
}

//对协方差矩阵和模型权值进行初始化;
void GMM::initParams()
{
	Eigen::MatrixXd distmat(n_data, m_n);
	//Eigen::MatrixXd::Index* minIndex = new Eigen::MatrixXd::Index[n_data];
	for (int i = 0; i < n_data;i++)
	{
		distmat.row(i) = (m_data->row(i).replicate(m_n, 1) - (*m_miu)).
			rowwise().squaredNorm();
	}
	Eigen::VectorXi Max = Eigen::VectorXi::Zero(m_n);
	Eigen::VectorXi rowMax = Eigen::VectorXi::Zero(n_data);
	for (int i = 0; i < n_data;i++)
	{
		Eigen::MatrixXd::Index index;
		distmat.row(i).minCoeff(&index);
		rowMax(i) = index;
		Max(index)++;
	}
	//std::cout << Max;
	for (int i = 0; i < m_n;i++)
	{
		(*m_pai)(i) = Max(i) / double(n_data);
		//std::cout << (*m_pai)(i) << std::endl;
	}
	
	Eigen::MatrixXd* nData = new Eigen::MatrixXd[m_n];
	int *nIndex = new int[m_n];
	for (int i = 0; i < m_n;i++)
	{
		nData[i] = Eigen::MatrixXd::Matrix(Max(i),m_dim);
		nIndex[i] = 0;
	}
	for (int i = 0; i < n_data;i++)
	{
		nData[rowMax(i)].row(nIndex[rowMax(i)]) = m_data->row(i);
		nIndex[rowMax(i)]++;
	}
	//std::cout << (*(nData)) << std::endl;
	//std::cout<< (*(nData + 1)) << std::endl;
	for (int i = 0; i < m_n;i++)
	{
		cov(nData + i, m_omega[i]);
		//std::cout << *(m_omega[i]) << std::endl;
	}
	delete[]nData;
	delete[]nIndex;
	//delete[]minIndex;
}

void GMM::cov(Eigen::MatrixXd* in, Eigen::MatrixXd* out)
{
	Eigen::MatrixXd off = (*in) - in->eval().colwise().mean().replicate(in->rows(),1);
	//std::cout << (*in) << std::endl;
	(*out) = off.transpose()*off / (in->rows()-1);
}

void GMM::output()
{
	std::cout << "期望值:" << std::endl << *m_miu << std::endl;
	for (int i = 0; i < m_n;i++)
	{
		std::cout << i << "omega:" << std::endl;
		std::cout << (*(m_omega[i])) << std::endl;
	}
}

//返回n_data*m_n的矩阵
Eigen::MatrixXd* GMM::calcProb()
{
	Eigen::MatrixXd *Px = new Eigen::MatrixXd(n_data, m_n);
	for (int i = 0; i < m_n;i++)
	{
		Eigen::MatrixXd XShift = (*m_data) - m_miu->row(i).replicate(n_data,1);
		//预防奇异阵，在协方差矩阵对角加上一个小数;
		//*(m_omega[i]) = (*(m_omega[i])).eval() + Eigen::MatrixXd::Ones(m_dim, m_dim).diagonal().asDiagonal();
		for (int j = 0; j < m_dim; j++)
		{
			double tmp = (*(m_omega[i]))(j, j);
			(*(m_omega[i]))(j, j) = tmp + 1;
		}
		Eigen::MatrixXd omegaInv = (m_omega[i])->inverse();
		
		Eigen::MatrixXd tmpM = (XShift*omegaInv);
		tmpM = tmpM.eval().cwiseProduct(XShift);
		Eigen::VectorXd tmpV = tmpM.rowwise().sum();
		//int sz = tmpV.size();
		double coef = pow(2 * PI, -double(m_dim) / 2)*sqrt(omegaInv.determinant());
		Px->col(i) = ((-tmpV*0.5).array().exp()*coef).matrix();
		//Px->col(i) = ((tmpV*0.5).exp())*coef;
	}
	return Px;
}

void GMM::calculateProbility(Eigen::MatrixXd* data, Eigen::MatrixXd* probability, int n)
{
	int nData = data->rows();
	Eigen::MatrixXd Pk(nData, m_n);//数据在m_n个高斯混合模型的值;
	for (int i = 0; i < m_n;i++)
	{
		Eigen::MatrixXd XShift = (*data) - m_miu->row(i).replicate(nData, 1);
		for (int j = 0; j < m_dim; j++)
		{
			double tmp = (*(m_omega[i]))(j, j);
			(*(m_omega[i]))(j, j) = tmp + 1;
		}
		Eigen::MatrixXd omegaInv = (m_omega[i])->inverse();

		Eigen::MatrixXd tmpM = (XShift*omegaInv);
		tmpM = tmpM.eval().cwiseProduct(XShift);
		Eigen::VectorXd tmpV = tmpM.rowwise().sum();
		double coef = pow(2 * PI, -double(m_dim) / 2)*sqrt(omegaInv.determinant());
		Pk.col(i) = ((-tmpV*0.5).array().exp()*coef).matrix();
	}
	probability->col(n) = Pk.cwiseProduct(m_pai->replicate(nData, 1)).rowwise().sum();
}