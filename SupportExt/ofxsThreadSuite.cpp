/* ***** BEGIN LICENSE BLOCK *****
 * This file is part of openfx-supportext <https://github.com/devernay/openfx-supportext>,
 * Copyright (C) 2013-2017 INRIA
 *
 * openfx-supportext is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * openfx-supportext is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with openfx-supportext.  If not, see <http://www.gnu.org/licenses/gpl-2.0.html>
 * ***** END LICENSE BLOCK ***** */

/*
 * A plugin-side multithread suite.
 * Can be used in place of a faulty or missing host MultiThread Suite.
 *
 * This suite counts the number of running threads lauched by this suite only, and reports the number of free slots in multiThreadNumCPUs.
 *
 * The number of free slots is shared between all plugins of a multibundle.
 */

//#define DEBUG_STDOUT // output debug messages to stdout

#include "ofxsThreadSuite.h"
#include "ofxsMultiThread.h"

#include <cassert>
#include <vector>
#include <map>
#ifdef DEBUG_STDOUT
#include <iostream>
#define DBG(x) (x)
#else
#define DBG(x) (void)0
#endif

// use TinyThread 1.2 from https://gitorious.org/tinythread/tinythreadpp
// for portable C++11-like threads
#include "tinythread.h"
// use our version of fast_mutex.h, which has bug fixes
//#include "fast_mutex.h"

#include "ofxCore.h"
#include "ofxMultiThread.h"
#include "ofxsImageEffect.h"

using namespace tthread;
using std::map;
using std::vector;
#ifdef DEBUG_STDOUT
using std::cout;
using std::endl;
#endif

namespace {

OfxStatus multiThreadNumCPUs(unsigned int *nCPUs);

const unsigned nprocs = thread::hardware_concurrency();

mutex occupancyLock; // protects occupancy
unsigned occupancy = 0;

mutex threadIndexesLock; // protects threadIndexes
map<thread::id, unsigned int> threadIndexes;


struct ThreadArgs
{
    OfxThreadFunctionV1* func;
    unsigned int threadIndex;
    unsigned int threadMax;
    void *customArg;
    OfxStatus ret;
};

void
threadFunction(void *_args)
{
    ThreadArgs* args = (ThreadArgs*)_args;
    assert(args->threadIndex < args->threadMax);

    args->ret = kOfxStatOK;
    try {
        args->func(args->threadIndex, args->threadMax, args->customArg);
    } catch (const std::bad_alloc & ba) {
        args->ret = kOfxStatErrMemory;
    } catch (...) {
        args->ret = kOfxStatFailed;
    }
}

/**@brief Function to spawn SMP threads

 \arg func The function to call in each thread.
 \arg nThreads The number of threads to launch
 \arg customArg The paramter to pass to customArg of func in each thread.

 This function will spawn nThreads separate threads of computation (typically one per CPU)
 to allow something to perform symmetric multi processing. Each thread will call 'func' passing
 in the index of the thread and the number of threads actually launched.

 multiThread will not return until all the spawned threads have returned. It is up to the host
 how it waits for all the threads to return (busy wait, blocking, whatever).

 \e nThreads can be more than the value returned by multiThreadNumCPUs, however the threads will
 be limitted to the number of CPUs returned by multiThreadNumCPUs.

 This function cannot be called recursively.

 @returns
 - ::kOfxStatOK, the function func has executed and returned sucessfully
 - ::kOfxStatFailed, the threading function failed to launch
 - ::kOfxStatErrExists, failed in an attempt to call multiThread recursively,

 */
// Note that the thread indexes are from 0 to nThreads-1.
// http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#OfxMultiThreadSuiteV1_multiThread
OfxStatus multiThread(OfxThreadFunctionV1 func,
                      unsigned int nThreads,
                      void *customArg)
{
    if (!func) {
        return kOfxStatFailed;
    }

    // check if this is a spawned thread, if yes return kOfxStatErrExists
    {
        lock_guard<mutex> guard(threadIndexesLock);
        if ( threadIndexes.find( this_thread::get_id() ) != threadIndexes.end() ) {
            return kOfxStatErrExists;
        }
    }

    unsigned int maxConcurrentThread;
    OfxStatus st = multiThreadNumCPUs(&maxConcurrentThread);
    if (st != kOfxStatOK) {
        return st;
    }

    // from the documentation:
    // "nThreads can be more than the value returned by multiThreadNumCPUs, however
    // the threads will be limitted to the number of CPUs returned by multiThreadNumCPUs."

    if ( (nThreads == 1) || (maxConcurrentThread <= 1) ) {
        int retval;
        {
            lock_guard<mutex> guard(occupancyLock);
            ++occupancy;
        }
        try {
            for (unsigned int i = 0; i < nThreads; ++i) {
                func(i, nThreads, customArg);
            }

            retval = kOfxStatOK;
        } catch (...) {
            retval = kOfxStatFailed;
        }
        {
            lock_guard<mutex> guard(occupancyLock);
            --occupancy;
        }
        return retval;
    }

    // at most maxConcurrentThread should be running at the same time
    vector<thread*> threads(nThreads, NULL);
    vector<thread::id> threadIDs(nThreads);
    vector<ThreadArgs> threadArgs(nThreads);
    for (unsigned int i = 0; i < nThreads; ++i) {
        threadArgs[i].func = func;
        threadArgs[i].threadIndex = i;
        threadArgs[i].threadMax = nThreads;
        threadArgs[i].customArg = customArg;
        threadArgs[i].ret = kOfxStatFailed;
    }
    unsigned int i = 0; // index of next thread to launch
    unsigned int running = 0; // number of running threads
    unsigned int j = 0; // index of first running thread. all threads before this one are finished running
    while (j < nThreads) {
        // have no more than maxConcurrentThread threads launched at the same time
        int threadsStarted = 0;
        {
            lock_guard<mutex> guard(threadIndexesLock);

            while (i < nThreads && running < maxConcurrentThread) {
                threads[i] = new thread(threadFunction, &threadArgs[i]);
                threadIDs[i] = threads[i]->get_id();
                assert( threadIndexes.find(threadIDs[i]) == threadIndexes.end() );
                threadIndexes[threadIDs[i]] = threadArgs[i].threadIndex;
                ++i;
                ++running;
                ++threadsStarted;
            }
        }

        ///We just started threadsStarted threads
        {
            lock_guard<mutex> guard(occupancyLock);
            occupancy += threadsStarted;
        }

        // now we've got at most maxConcurrentThread running. wait for each thread and launch a new one
        threads[j]->join();
        assert( !threads[j]->joinable() );
        {
            lock_guard<mutex> guard(threadIndexesLock);
            map<thread::id, unsigned int>::iterator it = threadIndexes.find(threadIDs[j]);
            assert( it != threadIndexes.end() );
            if ( it != threadIndexes.end() ) {
                threadIndexes.erase(it);
            }
        }
        delete threads[j];
        threads[j] = NULL;
        threadIDs[j] = thread::id();
        ++j;
        --running;

        // We just stopped 1 thread
        {
            lock_guard<mutex> guard(occupancyLock);
            --occupancy;
        }
    }
    assert(running == 0);

    // check the return status of each thread, return the first error found
    for (unsigned int i = 0; i < nThreads; ++i) {
        OfxStatus stat = threadArgs[i].ret;
        if (stat != kOfxStatOK) {
            return stat;
        }
    }

    return kOfxStatOK;
}

/**@brief Function which indicates the number of CPUs available for SMP processing

 \arg nCPUs pointer to an integer where the result is returned

 This value may be less than the actual number of CPUs on a machine, as the host may reserve other CPUs for itself.

 @returns
 - ::kOfxStatOK, all was OK and the maximum number of threads is in nThreads.
 - ::kOfxStatFailed, the function failed to get the number of CPUs
 */
// http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#OfxMultiThreadSuiteV1_multiThreadNumCPUs
OfxStatus multiThreadNumCPUs(unsigned int *nCPUs)
{
    lock_guard<mutex> guard(occupancyLock);
    *nCPUs = occupancy >= nprocs ? 1 : (nprocs - occupancy);
    DBG(std::cout << "numCPUs=" << *nCPUs << endl);
    return kOfxStatOK;
}

/**@brief Function which indicates the index of the current thread

 \arg threadIndex  pointer to an integer where the result is returned

 This function returns the thread index, which is the same as the \e threadIndex argument passed to the ::OfxThreadFunctionV1.

 If there are no threads currently spawned, then this function will set threadIndex to 0

 @returns
 - ::kOfxStatOK, all was OK and the maximum number of threads is in nThreads.
 - ::kOfxStatFailed, the function failed to return an index
 */
// http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#OfxMultiThreadSuiteV1_multiThreadIndex
// Note that the thread indexes are from 0 to nThreads-1, so a return value of 0 does not mean that it's not a spawned thread
// (use multiThreadIsSpawnedThread() to check if it's a spawned thread)
OfxStatus multiThreadIndex(unsigned int *threadIndex)
{
    if (!threadIndex) {
        return kOfxStatFailed;
    }

    lock_guard<mutex> guard(threadIndexesLock);
    map<thread::id, unsigned int>::const_iterator it = threadIndexes.find( this_thread::get_id() );
    if ( it != threadIndexes.end() ) {
        *threadIndex = it->second;
    } else {
        *threadIndex = 0;
    }

    return kOfxStatOK;
}

/**@brief Function to enquire if the calling thread was spawned by multiThread

 @returns
 - 0 if the thread is not one spawned by multiThread
 - 1 if the thread was spawned by multiThread
 */
// http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#OfxMultiThreadSuiteV1_multiThreadIsSpawnedThread
int multiThreadIsSpawnedThread(void)
{
    lock_guard<mutex> guard(threadIndexesLock);
    return threadIndexes.find( this_thread::get_id() ) != threadIndexes.end();
}

/** @brief Create a mutex

 \arg mutex - where the new handle is returned
 \arg count - initial lock count on the mutex. This can be negative.

 Creates a new mutex with lockCount locks on the mutex intially set.

 @returns
 - kOfxStatOK - mutex is now valid and ready to go
 */
// http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#OfxMultiThreadSuiteV1_mutexCreate
OfxStatus mutexCreate(OfxMutexHandle *mutex, int lockCount)
{
    if (!mutex) {
        return kOfxStatFailed;
    }

    // suite functions should not throw
    try {
        recursive_mutex* m = new recursive_mutex();
        for (int i = 0; i < lockCount; ++i) {
            m->lock();
        }
        *mutex = (OfxMutexHandle)(m);

        return kOfxStatOK;
    } catch (std::bad_alloc) {
        DBG(cout << "mutexCreate(): memory error.\n");

        return kOfxStatErrMemory;
    } catch (const std::exception & e) {
        DBG(cout << "mutexCreate(): " << e.what() << endl);

        return kOfxStatErrUnknown;
    } catch (...) {
        DBG(cout << "mutexCreate(): unknown error.\n");

        return kOfxStatErrUnknown;
    }
}

/** @brief Destroy a mutex

 Destroys a mutex intially created by mutexCreate.

 @returns
 - kOfxStatOK - if it destroyed the mutex
 - kOfxStatErrBadHandle - if the handle was bad
 */
// http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#OfxMultiThreadSuiteV1_mutexDestroy
OfxStatus mutexDestroy(const OfxMutexHandle mutex)
{
    if (mutex == 0) {
        return kOfxStatErrBadHandle;
    }
    // suite functions should not throw
    try {
        delete reinterpret_cast<recursive_mutex*>(mutex);

        return kOfxStatOK;
    } catch (std::bad_alloc) {
        DBG(cout << "mutexDestroy(): memory error.\n");

        return kOfxStatErrMemory;
    } catch (const std::exception & e) {
        DBG(cout << "mutexDestroy(): " << e.what() << endl);

        return kOfxStatErrUnknown;
    } catch (...) {
        DBG(cout << "mutexDestroy(): unknown error.\n");

        return kOfxStatErrUnknown;
    }
}

/** @brief Blocking lock on the mutex

 This trys to lock a mutex and blocks the thread it is in until the lock suceeds.

 A sucessful lock causes the mutex's lock count to be increased by one and to block any other calls to lock the mutex until it is unlocked.

 @returns
 - kOfxStatOK - if it got the lock
 - kOfxStatErrBadHandle - if the handle was bad
 */
// http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#OfxMultiThreadSuiteV1_mutexLock
OfxStatus mutexLock(const OfxMutexHandle mutex)
{
    if (mutex == 0) {
        return kOfxStatErrBadHandle;
    }
    // suite functions should not throw
    try {
        reinterpret_cast<recursive_mutex*>(mutex)->lock();

        return kOfxStatOK;
    } catch (std::bad_alloc) {
        DBG(cout << "mutexLock(): memory error.\n");

        return kOfxStatErrMemory;
    } catch (const std::exception & e) {
        DBG(cout << "mutexLock(): " << e.what() << endl);

        return kOfxStatErrUnknown;
    } catch (...) {
        DBG(cout << "mutexLock(): unknown error.\n");

        return kOfxStatErrUnknown;
    }
}

/** @brief Unlock the mutex

 This  unlocks a mutex. Unlocking a mutex decreases its lock count by one.

 @returns
 - kOfxStatOK if it released the lock
 - kOfxStatErrBadHandle if the handle was bad
 */
// http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#OfxMultiThreadSuiteV1_mutexUnLock
OfxStatus mutexUnLock(const OfxMutexHandle mutex)
{
    if (mutex == 0) {
        return kOfxStatErrBadHandle;
    }
    // suite functions should not throw
    try {
        reinterpret_cast<recursive_mutex*>(mutex)->unlock();

        return kOfxStatOK;
    } catch (std::bad_alloc) {
        DBG(cout << "mutexUnLock(): memory error.\n");

        return kOfxStatErrMemory;
    } catch (const std::exception & e) {
        DBG(cout << "mutexUnLock(): " << e.what() << endl);

        return kOfxStatErrUnknown;
    } catch (...) {
        DBG(cout << "mutexUnLock(): unknown error.\n");

        return kOfxStatErrUnknown;
    }
}

/** @brief Non blocking attempt to lock the mutex

 This attempts to lock a mutex, if it cannot, it returns and says so, rather than blocking.

 A sucessful lock causes the mutex's lock count to be increased by one, if the lock did not suceed, the call returns immediately and the lock count remains unchanged.

 @returns
 - kOfxStatOK - if it got the lock
 - kOfxStatFailed - if it did not get the lock
 - kOfxStatErrBadHandle - if the handle was bad
 */
// http://openfx.sourceforge.net/Documentation/1.3/ofxProgrammingReference.html#OfxMultiThreadSuiteV1_mutexTryLock
OfxStatus mutexTryLock(const OfxMutexHandle mutex)
{
    if (mutex == 0) {
        return kOfxStatErrBadHandle;
    }
    // suite functions should not throw
    try {
        if ( reinterpret_cast<recursive_mutex*>(mutex)->try_lock() ) {
            return kOfxStatOK;
        } else {
            return kOfxStatFailed;
        }
    } catch (std::bad_alloc) {
        DBG(cout << "mutexTryLock(): memory error.\n");

        return kOfxStatErrMemory;
    } catch (const std::exception & e) {
        DBG(cout << "mutexTryLock(): " << e.what() << endl);

        return kOfxStatErrUnknown;
    } catch (...) {
        DBG(cout << "mutexTryLock(): unknown error.\n");

        return kOfxStatErrUnknown;
    }
}

OfxMultiThreadSuiteV1 threadSuite = {
    multiThread,
    multiThreadNumCPUs,
    multiThreadIndex,
    multiThreadIsSpawnedThread,
    mutexCreate,
    mutexDestroy,
    mutexLock,
    mutexUnLock,
    mutexTryLock
};

OfxMultiThreadSuiteV1 mutexSuite = {
    NULL,
    NULL,
    NULL,
    NULL,
    mutexCreate,
    mutexDestroy,
    mutexLock,
    mutexUnLock,
    mutexTryLock
};

} // namespace {

namespace OFX {

extern ImageEffectHostDescription gHostDescription;
namespace Private {
    /** @brief Pointer to the plugin-side threading suite, can be used to replace gThreadSuite */
    OfxMultiThreadSuiteV1 *gPluginThreadSuite = &threadSuite;
    extern int gLoadCount;
    extern OfxMultiThreadSuiteV1 *gThreadSuite;
}

void ofxsThreadSuiteCheck()
{
    if (Private::gThreadSuite == NULL) {
        DBG(cout << "ofxsThreadSuiteCheck(): host has no thread suite.\n");
    } else if (Private::gThreadSuite->multiThread == NULL) {
        DBG(cout << "ofxsThreadSuiteCheck(): multiThread is NULL.\n");
    } else if (Private::gThreadSuite->multiThreadNumCPUs == NULL) {
        DBG(cout << "ofxsThreadSuiteCheck(): multiThreadNumCPUs is NULL.\n");
    } else if (Private::gThreadSuite->multiThreadIndex == NULL) {
        DBG(cout << "ofxsThreadSuiteCheck(): multiThreadIndex is NULL.\n");
    } else if (Private::gThreadSuite->multiThreadIsSpawnedThread == NULL) {
        DBG(cout << "ofxsThreadSuiteCheck(): multiThreadIsSpawnedThread is NULL.\n");
    } else if (Private::gThreadSuite->mutexCreate == NULL) {
        DBG(cout << "ofxsThreadSuiteCheck(): mutexCreate is NULL.\n");
    } else if (Private::gThreadSuite->mutexDestroy == NULL) {
        DBG(cout << "ofxsThreadSuiteCheck(): mutexDestroy is NULL.\n");
    } else if (Private::gThreadSuite->mutexLock == NULL) {
        DBG(cout << "ofxsThreadSuiteCheck(): mutexLock is NULL.\n");
    } else if (Private::gThreadSuite->mutexUnLock == NULL) {
        DBG(cout << "ofxsThreadSuiteCheck(): mutexUnLock is NULL.\n");
    } else if (Private::gThreadSuite->mutexTryLock == NULL) {
        DBG(cout << "ofxsThreadSuiteCheck(): mutexTryLock is NULL.\n");
    }
    // do it even if gLoadCount > 1. the load action is never multithreaded anyway
    if (Private::gThreadSuite != &threadSuite &&
        (Private::gThreadSuite == NULL ||
         Private::gThreadSuite->multiThread == NULL ||
         Private::gThreadSuite->multiThreadNumCPUs == NULL ||
         Private::gThreadSuite->multiThreadIndex == NULL ||
         Private::gThreadSuite->multiThreadIsSpawnedThread == NULL ||
         gHostDescription.hostName.compare(0, 14, "DaVinciResolve") == 0)) { // Resolve has a dummy MT Suite, use our own
        DBG(cout << "ofxsThreadSuiteCheck(): replacing host suite.\n");
        Private::gThreadSuite = &threadSuite;
    }
    if (Private::gThreadSuite != &mutexSuite &&
        (Private::gThreadSuite->mutexCreate == NULL ||
         Private::gThreadSuite->mutexDestroy == NULL ||
         Private::gThreadSuite->mutexLock == NULL ||
         Private::gThreadSuite->mutexUnLock == NULL ||
         Private::gThreadSuite->mutexTryLock == NULL ||
         gHostDescription.hostName.compare(0, 22, "com.sony.Catalyst.Edit") == 0)) { // Sony Catalyst Edit (as of version  2015.15) misses the mutex functions
        DBG(cout << "ofxsThreadSuiteCheck(): replacing host mutex suite.\n");
        mutexSuite.multiThread = Private::gThreadSuite->multiThread;
        mutexSuite.multiThreadNumCPUs = Private::gThreadSuite->multiThreadNumCPUs;
        mutexSuite.multiThreadIndex = Private::gThreadSuite->multiThreadIndex;
        mutexSuite.multiThreadIsSpawnedThread = Private::gThreadSuite->multiThreadIsSpawnedThread;
        Private::gThreadSuite = &mutexSuite;
    }
}

} // namespace OFX


