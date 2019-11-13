/*
 * The information in this file is
 * Copyright(c) 2007 Ball Aerospace & Technologies Corporation
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from   
 * http://www.gnu.org/licenses/lgpl.html
 */

#ifndef MULTITHREADEDALGORITHM_H
#define MULTITHREADEDALGORITHM_H

#include <string>
#include <vector>
#include <math.h>
#include "bmutex.h"
#include "bthread.h"
#include "bthread_signal.h"
#include "DMutex.h"
#include "EnumWrapper.h"
#include "MessageLogResource.h"

#include <numeric>
#include <algorithm>
#include <sstream>
#include "DesktopServices.h"

class Progress;
class MessageLogMgr;

/**
 * MTA is the Multi-Threaded Algorithm namespace.
 */
namespace mta // Multi-Threaded Algorithm
{

/**
* Calculate the required number of threads for the data to be processed.
* Using this prevents some oddities involving empty worker threads.
*
* @param dataSize
*        Size of the data to be processed by multiple threads.
* @return The number of threads required to process the data.
*/
unsigned int getNumRequiredThreads(unsigned int dataSize);

/**
 * Represents the result of algorithm execution.
 */
enum ResultEnum { SUCCESS, FAILURE, ABORT };

/**
 * @EnumWrapper mta::ResultEnum.
 */
typedef EnumWrapper<ResultEnum> Result;

/**
 * An action that can be run.
 */
class ThreadCommand
{
public:
   /**
    * Perform the action.
    */
   virtual void run() = 0;
};

/**
 * Report progress and errors from a thread.
 */
class ThreadReporter
{
public:
   /**
    * Send a progress report form a thread.
    *
    * @param threadIndex
    *        ID number for the current thread.
    * @param percentDone
    *        How much of this thread's progress has completed.
    * @return The result of the operation.
    */
   virtual Result reportProgress(int threadIndex, int percentDone) = 0;

   /**
    * Indicate that a thread has completed execution.
    *
    * @param threadIndex
    *        ID number for the current thread.
    * @return The result of the operation.
    */
   virtual Result reportCompletion(int threadIndex) = 0;

   /**
    * Report an error.
    *
    * @param errorText
    *        Description of the error.
    * @return The result of the operation.
    */
   virtual Result reportError(std::string errorText) = 0;

   /**
    * Access the most recent error.
    *
    * @return The description of the most recent error.
    */
   virtual std::string getErrorText() const = 0;

   /**
    * Access the progress of a thread.
    *
    * @param threadIndex
    *        The ID of the thread to check.
    * @return Percent complete for the thread.
    */
   virtual int getProgress(int threadIndex) const = 0;

   /**
    * Perform an action in the main thread.
    *
    * @param command
    *        The action to run in the main thread.
    */
   virtual void runInMainThread(ThreadCommand& command) = 0;
};

/**
 * Communicates between a processing thread and the main thread.
 */
class MultiThreadReporter : public ThreadReporter
{
public:
   /**
    * Messages sent between threads.
    */
   enum ReportTypeEnum
   {
      THREAD_NO_REPORT = 0x0,
      THREAD_PROGRESS = 0x1,
      THREAD_ERROR = 0x2,
      THREAD_COMPLETE = 0x4,
      THREAD_WORK = 0x8
   };

   /**
    * @EnumWrapper MultiThreadReporter::ReportTypeEnum.
    */
   typedef EnumWrapper<ReportTypeEnum> ReportType;

   /**
    * Constructor.
    *
    * @param threadCount
    *        The number of threads to executre.
    * @param pResult
    *        The result of execution will be stored here.
    * @param mutexA
    *        The first mutex used to coordinate the threads.
    * @param signalA
    *        The first signal used to coordinate the threads.
    * @param mutexB
    *        The second mutex used to coordinate the threads.
    * @param signalB
    *        The second signal used to coordinate the threads.
    */
   MultiThreadReporter(int threadCount, Result* pResult, BMutex& mutexA, BThreadSignal& signalA, BMutex& mutexB,
      BThreadSignal& signalB);

   /**
    * Copy constructor.
    *
    * @param reporter
    *        The other
    */
   MultiThreadReporter(MultiThreadReporter& reporter);

   /**
    * @copydoc ThreadReporter::reportProgress()
    */
   Result reportProgress(int threadIndex, int percentDone);

   /**
    * @copydoc ThreadReporter::reportCompletion()
    */
   Result reportCompletion(int threadIndex);

   /**
    * @copydoc ThreadReporter::reportError()
    */
   Result reportError(std::string errorText);

   /**
    * Get the current thread's progress.
    *
    * @return Percent complete for the current thread.
    */
   int getProgress() const;

   /**
    * @copydoc ThreadReporter::getProgress
    */
   int getProgress(int threadIndex) const;
   
   /**
    * @copydoc ThreadReporter::getErrorText()
    */
   std::string getErrorText() const;
   
   /**
    * @copydoc ThreadReporter::runInMainThread()
    */
   void runInMainThread(ThreadCommand& command);

   /**
    * Set the type of report.
    *
    * @param type
    *        The report type.
    */
   void setReportType(ReportType type);

   /**
    * Access the report type.
    *
    * @return The report type.
    */
   unsigned int getReportType() const;
   
   /**
    * Access the current thread command.
    *
    * @return The current thread command.
    */
   ThreadCommand* getThreadCommand();

private:
   BMutex& mMutexA;
   BThreadSignal& mSignalA;
   BMutex& mMutexB;
   BThreadSignal& mSignalB;
   Result* mpResult;
   std::vector<int> mThreadProgress;
   std::string mErrorMessage;
   unsigned int mReportType;
   ThreadCommand* mpThreadCommand;
   mutable DMutex mReporterMutex;
   mutable DMutex mSignalMutex;

   Result signalMainThread(ThreadCommand& reportStatus, ReportType type);
};

// this pragma shushes a compiler warning regarding the initialization
// of the mThreadHandle with 'this'
#pragma warning (disable: 4355)

/**
 * Base class for an algorithm thread.
 */
class AlgorithmThread : public ThreadCommand
{
public:
   /**
    * Constructor.
    *
    * @param threadIndex
    *        The ID of this thread.
    * @param reporter
    *        Used to report status to the main thread.
    */
   AlgorithmThread(int threadIndex, ThreadReporter& reporter) : 
      mpAlgorithmMutex(NULL),
      mReporter(reporter), 
      mThreadHandle(static_cast<void*>(this),  reinterpret_cast<void*>(AlgorithmThread::threadFunction)), 
      mThreadIndex(threadIndex) {}

   /**
    * Copy constructor.
    *
    * @param thread
    *        The other
    */
   AlgorithmThread(const AlgorithmThread& thread) : 
      mpAlgorithmMutex(thread.mpAlgorithmMutex),
      mReporter(thread.mReporter), 
      mThreadHandle(static_cast<void*>(this),  reinterpret_cast<void*>(AlgorithmThread::threadFunction)),
      mThreadIndex(thread.mThreadIndex) {}

   /**
    * The function executed by the underlying threading system.
    *
    * @param pThreadData
    *        The data for the thread being executed.
    */
   static void threadFunction(AlgorithmThread* pThreadData);

   /**
    * Execute the thread.
    */
   virtual void run() = 0;

   /**
    * Launch the thread.
    *
    * @return False if there was an error.
    */
   bool launch();

   /**
    * Wait for thread compltion.
    *
    * @return False if there was an error.
    */
   bool wait();

   /**
    * Perform an action in the main thread.
    *
    * @param command
    *        The action to run in the main thread.
    */
   void runInMainThread(ThreadCommand& command);

   /**
    * Set the mutex which synchronizes the threads in an algorithm cluster.
    *
    * This should be the same object for all threads in the algorithm cluster.
    *
    * @param pMutex
    *        The mutex to synchronize.
    */
   void setAlgorithmMutex(DMutex* pMutex);

   /**
    * Wait to begin thread execution.
    *
    * Synchronizes start-up among all threads in an algorithm cluster.
    *
    * @see setAlgorithmMutex()
    */
   void waitForAlgorithmLoop();

   /**
    * Represents a range in integers.
    */
   class Range
   {
   public:
      /**
       * Creates a Range.
       */
      Range() :
         mFirst(0),
         mLast(0)
      {
      }

      /**
       * Thread the range as a percentage.
       *
       * @return The percentage representation of the range.
       */
      int computePercent(int index)
      {
         return (100 * (index - mFirst)) / (mLast - mFirst + 1);
      }

      /**
       * The lower end of the range.
       */
      int mFirst;

      /**
       * The upper end of the range.
       */
      int mLast;
   };

protected:
   /**
    * Calculate the range of values for which a thread is responsible.
    *
    * @param threadCount
    *        The total number of threads in an algorithm cluster.
    * @param dataSize
    *        The total number of items which need to be processed.
    * @return The range of items which this thread will process.
    */
   Range getThreadRange(int threadCount, int dataSize) const;

   /**
    * Get the id of this thread.
    *
    * @return The id of this thread.
    */
   int getThreadIndex() const;

   /**
    * Get the object which this thread can use to report status.
    *
    * @return The object used to report status.
    */
   ThreadReporter& getReporter() const;

private:
   DMutex* mpAlgorithmMutex;
   ThreadReporter& mReporter;
   BThread mThreadHandle;
   int mThreadIndex;
};

/** \page multithreadedhowto Writing a multi-threaded algorithm
 * Use this template to make a thread class.
 * @code
 * class MyAlgorithmThread : public AlgorithmThread
 * {
 * public:
 *    MyAlgorithmThread(const MyAlgInput& input, int threadCount, int threadIndex, ThreadReporter& reporter) : 
 *       AlgorithmThread(threadIndex, reporter) 
 *    { 
 *       // parse 'input' for 'threadIndex' into member data 
 *    }
 *    // required by STL functions used in MultiThreadedAlgorithm
 *    MyAlgorithmThread& operator=(const MyAlgorithmThread& thread)
 *    {
 *       *this = thread;
 *       return *this;
 *    }
 *    // this function is called in the separate thread and does the real work
 *    void run();
 * private:
 *    // put per-thread information into member data here
 * };
 * @endcode
 */

/**
 * Base progress reporting class.
 */
class ProgressReporter
{
public:
   /**
    * Report progress percent.
    *
    * @param percent
    *        The percent completion.
    */
   virtual void reportProgress(int percent) = 0;

   /**
    * Report an error.
    *
    * @param text
    *        The error message.
    */
   virtual void reportError(const std::string& text) = 0;
};

/**
 * Use this derived class to report to a progress object.
 */
class ProgressObjectReporter : public ProgressReporter
{
public:
   /**
    * Constructor.
    *
    * @param baseMessage
    *        The base message used when constructing a message.
    * @param pProgress
    *        Progress object to report. If this is NULL, no Progress will receive reports.
    */
   ProgressObjectReporter(std::string baseMessage, Progress* pProgress) :
      mMessage(baseMessage),
      mpProgress(pProgress)
   {
   }

   /**
    * @copydoc ProgressReporter::reportProgress()
    */
   void reportProgress(int percent);

   /**
    * @copydoc ProgressReporter::reportError()
    */
   void reportError(const std::string& text);

private:
   std::string mMessage;
   Progress* mpProgress;
};

/**
 * Use this derived class to report to the status bar.
 */
class StatusBarReporter : public ProgressReporter
{
public:
   /**
    * Constructor.
    *
    * @param baseMessage
    *        The base message used when constructing a message.
    * @param component
    *        Message log component.
    * @param key
    *        Message log key.
    */
   StatusBarReporter(std::string baseMessage, const std::string& component, const std::string& key) :
      mMessage(baseMessage),
      mComponent(component),
      mKey(key)
   {}

   /**
    * @copydoc ProgressReporter::reportProgress()
    */
   void reportProgress(int percent)
   {
      std::stringstream buf;
      buf << mMessage << ": " << percent << "%";
      Service<DesktopServices>()->setStatusBarMessage(buf.str());
   }

   /**
    * @copydoc ProgressReporter::reportError()
    */
   void reportError(const std::string& text)
   {
      Service<DesktopServices>()->setStatusBarMessage(text);
      MessageResource msg("Error", mComponent, mKey);
      msg->addProperty("Message", text);
   }
private:
   std::string mMessage;
   const std::string& mComponent;
   const std::string& mKey;
};


/**
 * Use this class to report progress from a multi-phase algorithm.
 */
class MultiPhaseProgressReporter : public ProgressReporter
{
public:
   /**
    * Constructor.
    *
    * @param base
    *        ProgressReporter used for reports.
    * @param phaseWeights
    *        How much each phase contributes to the overall progress.
    */
   MultiPhaseProgressReporter(ProgressReporter& base, const std::vector<int>& phaseWeights) :
      mReporter(base), mPhaseWeights(phaseWeights), mCurrentPhase(0) {}

   /**
    * @copydoc ProgressReporter::reportProgress()
    */
   void reportProgress(int percent);

   /**
    * @copydoc ProgressReporter::reportError()
    */
   void reportError(const std::string& text);

   /**
    * Set the current algorithm phase.
    *
    * @param phase
    *        The phase the algorithm is executing.
    */
   void setCurrentPhase(int phase);

   /**
    * Access the current algorithm phase.
    *
    * @return The current phase.
    */
   int getCurrentPhase() const;

private:
   int convertPhaseProgressToTotalProgress(int phaseProgress);

   ProgressReporter& mReporter;
   std::vector<int> mPhaseWeights;
   int mCurrentPhase;
};

/**
 * An algorithm which distributes work between multiply threads. (SIMD)
 */
template<class AlgInput, class AlgOutput, class AlgThread>
class MultiThreadedAlgorithm
{
public:
   /**
    * Constructor.
    *
    * @param threadCount
    *        Number of threads to create.
    * @param input
    *        Algorithm input.
    * @param output
    *        Algorithm output.
    * @param pProgress
    *        Used to report progress and errors.
    */
   MultiThreadedAlgorithm(int threadCount, const AlgInput& input, AlgOutput& output, ProgressReporter* pProgress);

   /**
    * Destructor.
    */
   ~MultiThreadedAlgorithm();
   
   /**
    * Execute the algorithm.
    *
    * @return The result of the execution.
    */
   Result run();

   /**
    * The last error message.
    * If the result of run() is an error, this returns the error description.
    *
    * @return The last error message or an empty string if no error occured.
    */
   std::string getErrorText() const
   {
      return mErrorText;
   }

private:
   Result createThreads(int threadCount);
   Result startAllThreads();
   Result waitForThreadsToComplete();
   int processCurrentReports(int percentDone);
   int processReport(unsigned int currentType, int percentDone);
   Result compileResults();

   Result mCurrentStatus;
   const AlgInput& mInput;
   AlgOutput& mOutput;
   std::vector<AlgThread*> mThreads;
   MultiThreadReporter* mpThreadReporter;
   ProgressReporter* mpProgressReporter;
   DMutex mMutexA;
   DThreadSignal mSignalA;
   DMutex mMutexB;
   DThreadSignal mSignalB;
   std::string mErrorText;
};

template<class AlgInput, class AlgOutput, class AlgThread>
MultiThreadedAlgorithm<AlgInput, AlgOutput, AlgThread>::MultiThreadedAlgorithm(int threadCount,
   const AlgInput& algInput, AlgOutput& algOutput, ProgressReporter* pReporter) :
   mCurrentStatus(SUCCESS),
   mInput(algInput),
   mOutput(algOutput),
   mpThreadReporter(NULL),
   mpProgressReporter(pReporter)
{
   mpThreadReporter = new MultiThreadReporter(threadCount, &mCurrentStatus, mMutexA, mSignalA, mMutexB, mSignalB);
   createThreads(threadCount);
}

template<class AlgInput, class AlgOutput, class AlgThread>
MultiThreadedAlgorithm<AlgInput, AlgOutput, AlgThread>::~MultiThreadedAlgorithm()
{
   typename std::vector<AlgThread*>::iterator iter;
   for (iter = mThreads.begin(); iter != mThreads.end(); ++iter)
   {
      AlgThread* pThread = *iter;
      if (pThread != NULL)
      {
         delete pThread;
      }
   }

   mThreads.clear();
   delete mpThreadReporter;
}

template<class AlgInput, class AlgOutput, class AlgThread>
Result MultiThreadedAlgorithm<AlgInput, AlgOutput, AlgThread>::createThreads(int threadCount)
{
   int i;
   for (i = 0; i < threadCount; ++i)
   {
      AlgThread* pThread = NULL;
      pThread = new AlgThread(mInput, threadCount, i, *mpThreadReporter);
      if (pThread != NULL)
      {
         pThread->setAlgorithmMutex(&mMutexA);
         mThreads.push_back(pThread);
      }
   }
   return SUCCESS;
}

template<class AlgInput, class AlgOutput, class AlgThread>
Result MultiThreadedAlgorithm<AlgInput, AlgOutput, AlgThread>::startAllThreads()
{
   typename std::vector<AlgThread*>::iterator iter;

   mMutexA.MutexLock();
   mMutexB.MutexLock();

   for (iter = mThreads.begin(); iter != mThreads.end(); ++iter)
   {
      (*iter)->launch();
   }
   return SUCCESS;
}

template<class AlgInput, class AlgOutput, class AlgThread>
Result MultiThreadedAlgorithm<AlgInput, AlgOutput, AlgThread>::waitForThreadsToComplete()
{
   bool doneProcessing = false;
   int percentDone = 0;

   mMutexB.MutexUnlock();
   while (!doneProcessing)
   {
      mSignalA.ThreadSignalWait(&mMutexA);

      percentDone = processCurrentReports(percentDone);
      doneProcessing = (percentDone == 100 || mCurrentStatus != SUCCESS);
      if (doneProcessing)
      {
         mMutexA.MutexUnlock();

         typename std::vector<AlgThread*>::iterator iter;
         for (iter = mThreads.begin(); iter != mThreads.end(); ++iter)
         {
            (*iter)->wait();
         }
      }
   }

   return mCurrentStatus;
}

template<class AlgInput, class AlgOutput, class AlgThread>
int MultiThreadedAlgorithm<AlgInput, AlgOutput, AlgThread>::processCurrentReports(int percentDone)
{
   mMutexB.MutexLock();

   int type = mpThreadReporter->getReportType();
   unsigned int currentType = MultiThreadReporter::THREAD_WORK;
   while (currentType != 0)
   {
      if (type & currentType)
      {
         percentDone = processReport(currentType, percentDone);
      }
      currentType /= 2;
   }

   mpThreadReporter->setReportType(MultiThreadReporter::THREAD_NO_REPORT);

   mSignalB.ThreadSignalActivate();
   mMutexB.MutexUnlock();

   return percentDone;
}

template<class AlgInput, class AlgOutput, class AlgThread>
int MultiThreadedAlgorithm<AlgInput, AlgOutput, AlgThread>::processReport(unsigned int currentType, int percentDone)
{
   switch (currentType)
   {
      case MultiThreadReporter::THREAD_NO_REPORT:
         break;
      case MultiThreadReporter::THREAD_COMPLETE:
      case MultiThreadReporter::THREAD_PROGRESS:
         percentDone = mpThreadReporter->getProgress();
         if (mpProgressReporter != NULL)
         {
            mpProgressReporter->reportProgress(percentDone);
         }
         break;
      case MultiThreadReporter::THREAD_ERROR:
         if (mpProgressReporter != NULL)
         {
            mpProgressReporter->reportError(mpThreadReporter->getErrorText().c_str());
         }
         mErrorText = mpThreadReporter->getErrorText().c_str();
         mCurrentStatus = FAILURE;
         break;
      case MultiThreadReporter::THREAD_WORK:
         if (mpThreadReporter->getThreadCommand() != NULL)
         {
            mpThreadReporter->getThreadCommand()->run();
         }
         break;
      default:
         break;
   }

   return percentDone;
}

template<class AlgInput, class AlgOutput, class AlgThread>
Result MultiThreadedAlgorithm<AlgInput, AlgOutput, AlgThread>::compileResults()
{
   bool success = mOutput.compileOverallResults(mThreads);
   return ((success == true) ? SUCCESS : FAILURE);
}

template<class AlgInput, class AlgOutput, class AlgThread>
Result MultiThreadedAlgorithm<AlgInput, AlgOutput, AlgThread>::run()
{
   Result result = startAllThreads();
   if (result == SUCCESS)
   {
      result = waitForThreadsToComplete();
   }

   if (result == SUCCESS)
   {
      result = compileResults();
   }

   return result;
}

} // end namespace mta

#endif
Back to Top
